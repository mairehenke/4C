// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_active_strain.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <cmath>
#include <span>
#include <string_view>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /// create the passive material and make sure it is a 3D solid material
  std::shared_ptr<Mat::So3Material> create_passive_material(const int matid)
  {
    auto passive = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(matid));
    if (!passive)
      FOUR_C_THROW("MAT_ActiveStrain: the passive material {} must be a 3D solid material.", matid);
    return passive;
  }

  /// read an optional fiber field from the material input
  std::optional<Mat::PAR::ActiveStrain::FiberField> read_optional_fiber_field(
      const Core::IO::InputParameterContainer& parameters, const std::string& name)
  {
    const auto* field = parameters.get_if<Mat::PAR::ActiveStrain::FiberField>(name);
    if (field == nullptr) return std::nullopt;
    return *field;
  }

  /// evaluate a space-time function at the reference coordinates of the current evaluation point
  double evaluate_space_time_function(
      const int funct_id, const Mat::EvaluationContext<3>& context, const std::string_view purpose)
  {
    FOUR_C_ASSERT_ALWAYS(context.ref_coords != nullptr,
        "MAT_ActiveStrain: the space-time function for {} needs the reference coordinates of the "
        "evaluation point, which the element did not provide.",
        purpose);

    const double total_time = (context.total_time != nullptr) ? *context.total_time : 0.0;

    return Global::Problem::instance()
        ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct_id)
        .evaluate(std::span<const double>(context.ref_coords->data(), 3), total_time, 0);
  }

  /// normalize a fiber direction
  Core::LinAlg::Tensor<double, 3> normalized(
      Core::LinAlg::Tensor<double, 3> fiber, const std::string_view name)
  {
    const double norm = Core::LinAlg::norm2(fiber);
    if (norm < 1e-12) FOUR_C_THROW("MAT_ActiveStrain: {} must not be zero.", name);
    fiber *= 1.0 / norm;
    return fiber;
  }
}  // namespace


Mat::PAR::ActiveStrain::ActiveStrain(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      matid_passive_(matdata.parameters.get<int>("MATIDPASSIVE")),
      scalid_voltage_(matdata.parameters.get<int>("SCALIDVOLTAGE")),
      density_(matdata.parameters.get<double>("DENS")),
      beta1_(matdata.parameters.get<double>("BETA1")),
      beta2_(matdata.parameters.get<double>("BETA2")),
      voltage_threshold_(matdata.parameters.get<double>("VOLTHRESH")),
      alpha_fiber1_(matdata.parameters.get<double>("ALPHA_FIBER1")),
      alpha_fiber2_(matdata.parameters.get<double>("ALPHA_FIBER2")),
      alpha_funct_fiber1_(matdata.parameters.get<int>("ALPHAFUNCT_FIBER1")),
      alpha_funct_fiber2_(matdata.parameters.get<int>("ALPHAFUNCT_FIBER2")),
      source_activation_(matdata.parameters.get<int>("SOURCE_ACTIVATION")),
      kinematics_(matdata.parameters.get<ActiveStrainKinematics>("ACTIVE_STRAIN_TYPE")),
      fiber1_field_(read_optional_fiber_field(matdata.parameters, "FIBER1")),
      fiber2_field_(read_optional_fiber_field(matdata.parameters, "FIBER2"))
{
  if (source_activation_ < 0)
  {
    FOUR_C_THROW(
        "MAT_ActiveStrain: SOURCE_ACTIVATION must be 0 (coupled scalar field) or the id of a "
        "space-time function, got {}.",
        source_activation_);
  }

  if (fiber2_field_.has_value() && !fiber1_field_.has_value())
    FOUR_C_THROW("MAT_ActiveStrain: the fiber field FIBER2 requires the fiber field FIBER1.");

  if (!is_transversely_isotropic() && fiber1_field_.has_value() != fiber2_field_.has_value())
  {
    FOUR_C_THROW(
        "MAT_ActiveStrain: orthotropic active strain kinematics need either both fiber fields "
        "FIBER1 and FIBER2 or none of them.");
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ActiveStrain::create_material()
{
  return std::make_shared<Mat::ActiveStrain>(this);
}


Mat::ActiveStrainType Mat::ActiveStrainType::instance_;

Core::Communication::ParObject* Mat::ActiveStrainType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* active_strain = new Mat::ActiveStrain();
  active_strain->unpack(buffer);
  return active_strain;
}


Mat::ActiveStrain::ActiveStrain() : params_(nullptr), matpassive_(nullptr) {}

Mat::ActiveStrain::ActiveStrain(Mat::PAR::ActiveStrain* params)
    : params_(params), matpassive_(create_passive_material(params->matid_passive_))
{
}

void Mat::ActiveStrain::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  add_to_pack(data, unique_par_object_id());

  // pack material id
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // element fiber triad and Gauss point data
  add_to_pack(data, element_triad_);
  add_to_pack(data, voltage_);
  add_to_pack(data, activation_);

  // pack the passive material (not accessible during post processing)
  Core::Communication::PotentiallyUnusedBufferScope passive_scope{data};
  if (params_ != nullptr) matpassive_->pack(data);
}

void Mat::ActiveStrain::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  matpassive_ = nullptr;

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract matid and recover params_
  int matid = -1;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr &&
      Global::Problem::instance()->materials()->num() != 0)
  {
    const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
    Core::Mat::PAR::Parameter* mat =
        Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
    if (mat->type() == material_type())
      params_ = dynamic_cast<Mat::PAR::ActiveStrain*>(mat);
    else
      FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
          material_type());
  }

  // element fiber triad and Gauss point data
  extract_from_pack(buffer, element_triad_);
  extract_from_pack(buffer, voltage_);
  extract_from_pack(buffer, activation_);

  // recreate and unpack the passive material (not accessible during post processing)
  Core::Communication::PotentiallyUnusedBufferScope passive_scope{buffer};
  if (params_ != nullptr)
  {
    matpassive_ = create_passive_material(params_->matid_passive_);
    matpassive_->unpack(buffer);
  }
}

void Mat::ActiveStrain::setup(const int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // the fiber directions of the element are only needed if no fiber input fields are given
  if (!params_->has_fiber_fields())
  {
    const bool needs_fiber2 = !params_->is_transversely_isotropic();

    if (coord_system.has_value())
    {
      // CIR-AXI-RAD nomenclature: fiber 1 = AXI, fiber 2 = CIR, normal = RAD
      element_triad_ = make_fiber_triad(coord_system->element_system[1],
          needs_fiber2 ? std::optional{coord_system->element_system[2]} : std::nullopt);
    }
    else if (fibers.element_fibers.size() >= (needs_fiber2 ? 2u : 1u))
    {
      // FIBER nomenclature: FIBER1 (and FIBER2) of the element
      element_triad_ = make_fiber_triad(fibers.element_fibers[0],
          needs_fiber2 ? std::optional{fibers.element_fibers[1]} : std::nullopt);
    }
    else
    {
      FOUR_C_THROW(
          "MAT_ActiveStrain: {} required. Either define {} on the element (FIBER1{} or the "
          "CIR-AXI-RAD coordinate system) or provide the material input field{} FIBER1{}.",
          needs_fiber2 ? "two fiber directions are" : "one fiber direction is",
          needs_fiber2 ? "them" : "it", needs_fiber2 ? "/FIBER2" : "", needs_fiber2 ? "s" : "",
          needs_fiber2 ? " and FIBER2" : "");
    }
  }

  matpassive_->setup(numgp, fibers, coord_system);

  voltage_.assign(numgp, 0.0);
  activation_.assign(numgp, 0.0);
}

void Mat::ActiveStrain::post_setup(const Teuchos::ParameterList& params, const int eleGID)
{
  matpassive_->post_setup(params, eleGID);
}

void Mat::ActiveStrain::update() { matpassive_->update(); }

void Mat::ActiveStrain::reset_step() { matpassive_->reset_step(); }

void Mat::ActiveStrain::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  FOUR_C_ASSERT(defgrad != nullptr, "MAT_ActiveStrain needs the deformation gradient.");

  // (transmembrane) voltage and resulting activation at this Gauss point
  const double voltage = evaluate_voltage(params, context);
  const double gamma = evaluate_activation(voltage);
  voltage_[gp] = voltage;
  activation_[gp] = gamma;

  // multiplicative split F = Fe * Fa  ->  Fe = F * Fa^{-1}
  const FiberTriad triad = evaluate_fiber_triad(eleGID, context);
  const Core::LinAlg::Tensor<double, 3, 3> inv_defgrad_active =
      evaluate_inverse_active_deformation_gradient(gamma, triad, context);
  const Core::LinAlg::Tensor<double, 3, 3> defgrad_passive = *defgrad * inv_defgrad_active;

  // passive Green-Lagrange strain Ee = 1/2 (Fe^T Fe - I)
  const Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain_passive =
      Core::LinAlg::assume_symmetry(
          0.5 *
          (Core::LinAlg::transpose(defgrad_passive) * defgrad_passive -
              Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>)));

  // evaluate the passive material in the intermediate configuration
  Core::LinAlg::SymmetricTensor<double, 3, 3> stress_passive{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_passive{};
  matpassive_->evaluate(&defgrad_passive, glstrain_passive, params, context, stress_passive,
      cmat_passive, gp, eleGID);

  // pull back the 2nd Piola-Kirchhoff stress to the reference configuration:
  // S = Fa^{-1} Se Fa^{-T}
  stress = Core::LinAlg::assume_symmetry(
      inv_defgrad_active * stress_passive * Core::LinAlg::transpose(inv_defgrad_active));

  // pull back the tangent to the reference configuration:
  // C_ABCD = Fa^{-1}_Ai Fa^{-1}_Bj Fa^{-1}_Ck Fa^{-1}_Dl Ce_ijkl
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  cmat_view.update(1.0,
      Mat::pull_back_four_tensor(1.0, Core::LinAlg::make_matrix(inv_defgrad_active),
          Core::LinAlg::make_stress_like_voigt_view(cmat_passive)),
      0.0);
}

void Mat::ActiveStrain::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  matpassive_->register_output_data_names(names_and_size);
  names_and_size["voltage"] = 1;
  names_and_size["activation"] = 1;
}

bool Mat::ActiveStrain::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  const auto write_gp_values = [&](const std::vector<double>& gp_values)
  {
    for (std::size_t gp = 0; gp < gp_values.size(); ++gp) data(gp, 0) = gp_values[gp];
    return true;
  };

  if (name == "voltage") return write_gp_values(voltage_);
  if (name == "activation") return write_gp_values(activation_);

  return matpassive_->evaluate_output_data(name, data);
}

Mat::ActiveStrain::FiberTriad Mat::ActiveStrain::make_fiber_triad(
    const Core::LinAlg::Tensor<double, 3>& fiber1,
    const std::optional<Core::LinAlg::Tensor<double, 3>>& fiber2) const
{
  const Core::LinAlg::Tensor<double, 3> f1 = normalized(fiber1, "fiber direction 1");

  // transversely isotropic kinematics only need fiber 1
  if (params_->is_transversely_isotropic()) return {f1, {}, {}};

  FOUR_C_ASSERT(fiber2.has_value(), "Orthotropic active strain kinematics need fiber 2.");

  // orthogonalize fiber 2 with respect to fiber 1, the normal is the cross product of both
  Core::LinAlg::Tensor<double, 3> f2 = *fiber2 - Core::LinAlg::dot(f1, *fiber2) * f1;
  if (Core::LinAlg::norm2(f2) < 1e-12)
  {
    FOUR_C_THROW(
        "MAT_ActiveStrain: fiber directions 1 and 2 must not be zero or parallel, they need to "
        "span the plane of contraction.");
  }
  f2 = normalized(f2, "fiber direction 2");

  const Core::LinAlg::Tensor<double, 3> normal{{
      f1(1) * f2(2) - f1(2) * f2(1),
      f1(2) * f2(0) - f1(0) * f2(2),
      f1(0) * f2(1) - f1(1) * f2(0),
  }};

  return {f1, f2, normal};
}

Mat::ActiveStrain::FiberTriad Mat::ActiveStrain::evaluate_fiber_triad(
    const int eleGID, const EvaluationContext<3>& context) const
{
  if (!params_->has_fiber_fields()) return element_triad_;

  // fiber directions from the material input fields, possibly interpolated from nodal data
  FOUR_C_ASSERT_ALWAYS(context.xi != nullptr,
      "MAT_ActiveStrain: interpolating the fiber fields needs the parameter coordinates of the "
      "evaluation point, which the element did not provide.");

  const auto fiber1 = params_->fiber1_field_->interpolate(eleGID, context.xi->as_span(), "FIBER1");

  if (params_->is_transversely_isotropic()) return make_fiber_triad(fiber1, std::nullopt);

  return make_fiber_triad(
      fiber1, params_->fiber2_field_->interpolate(eleGID, context.xi->as_span(), "FIBER2"));
}

double Mat::ActiveStrain::evaluate_voltage(
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context) const
{
  // voltage prescribed by a space-time function (e.g. a traveling activation wave)
  if (params_->source_activation_ > 0)
    return evaluate_space_time_function(params_->source_activation_, context, "SOURCE_ACTIVATION");

  // voltage from the coupled scalar field, interpolated to the current Gauss point by the element
  if (!params.isParameter("scalars"))
  {
    FOUR_C_THROW(
        "MAT_ActiveStrain with SOURCE_ACTIVATION 0 requires the coupled scalar field, which the "
        "element did not provide. Use a solid-scatra element or set SOURCE_ACTIVATION to the id of "
        "a space-time function.");
  }
  const auto scalars = params.get<std::shared_ptr<std::vector<double>>>("scalars");
  const int num_scalars = (scalars != nullptr) ? static_cast<int>(scalars->size()) : 0;
  if (params_->scalid_voltage_ < 1 || params_->scalid_voltage_ > num_scalars)
  {
    FOUR_C_THROW("MAT_ActiveStrain: SCALIDVOLTAGE {} is out of range, {} scalar(s) are available.",
        params_->scalid_voltage_, num_scalars);
  }

  return (*scalars)[params_->scalid_voltage_ - 1];
}

double Mat::ActiveStrain::evaluate_activation(const double voltage) const
{
  if (voltage <= params_->voltage_threshold_) return 0.0;

  const double excess_voltage = voltage - params_->voltage_threshold_;
  return (1.0 - std::exp(-params_->beta1_ * excess_voltage)) *
         (1.0 - std::exp(-params_->beta2_ * excess_voltage));
}

Core::LinAlg::Tensor<double, 3, 3> Mat::ActiveStrain::evaluate_inverse_active_deformation_gradient(
    const double gamma, const FiberTriad& triad, const EvaluationContext<3>& context) const
{
  const auto& [fiber1, fiber2, normal] = triad;
  const auto identity =
      Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);

  // optional modulation of the contraction intensity in space and time
  const double modulation1 =
      (params_->alpha_funct_fiber1_ > 0)
          ? evaluate_space_time_function(params_->alpha_funct_fiber1_, context, "ALPHAFUNCT_FIBER1")
          : 1.0;
  const double gamma1 = gamma * params_->alpha_fiber1_ * modulation1;
  if (gamma1 >= 1.0)
  {
    FOUR_C_THROW(
        "MAT_ActiveStrain: the active contraction along fiber 1 (gamma1 = {}) must stay below 1, "
        "otherwise the active deformation gradient is singular.",
        gamma1);
  }

  if (params_->is_transversely_isotropic())
  {
    // Fa = (1-gamma1) f1(x)f1 + (1-gamma1)^{-1/2} (I - f1(x)f1), det(Fa) = 1
    const double transverse_stretch = std::sqrt(1.0 - gamma1);
    return transverse_stretch * identity +
           (1.0 / (1.0 - gamma1) - transverse_stretch) * Core::LinAlg::dyadic(fiber1, fiber1);
  }

  const double modulation2 =
      (params_->alpha_funct_fiber2_ > 0)
          ? evaluate_space_time_function(params_->alpha_funct_fiber2_, context, "ALPHAFUNCT_FIBER2")
          : 1.0;
  const double gamma2 = gamma * params_->alpha_fiber2_ * modulation2;
  if (gamma2 >= 1.0)
  {
    FOUR_C_THROW(
        "MAT_ActiveStrain: the active contraction along fiber 2 (gamma2 = {}) must stay below 1, "
        "otherwise the active deformation gradient is singular.",
        gamma2);
  }

  // contraction in normal direction such that the active deformation is isochoric, det(Fa) = 1
  const double gamma_normal =
      (1.0 - (1.0 - gamma1) * (1.0 - gamma2)) / ((1.0 - gamma1) * (1.0 - gamma2));

  // Fa = I - gamma1 f1(x)f1 - gamma2 f2(x)f2 + gamma_n n(x)n; its inverse for an orthonormal triad
  return identity + gamma1 / (1.0 - gamma1) * Core::LinAlg::dyadic(fiber1, fiber1) +
         gamma2 / (1.0 - gamma2) * Core::LinAlg::dyadic(fiber2, fiber2) -
         gamma_normal / (1.0 + gamma_normal) * Core::LinAlg::dyadic(normal, normal);
}

FOUR_C_NAMESPACE_CLOSE
