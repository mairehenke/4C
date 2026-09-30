// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ACTIVE_STRAIN_HPP
#define FOUR_C_MAT_ACTIVE_STRAIN_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class ActiveStrain;

  namespace PAR
  {
    /// parameter class for the 3D active strain material
    class ActiveStrain : public Core::Mat::PAR::Parameter
    {
      friend class Mat::ActiveStrain;

     public:
      /// kinematics of the active deformation gradient
      enum class ActiveStrainKinematics : std::uint8_t
      {
        /// two fibers contract, the normal direction compensates the volume (default)
        orthotropic,
        /// one fiber contracts, the transverse plane expands isotropically
        transversely_isotropic,
      };

      /// fiber direction field that may be constant, element-wise or interpolated from nodes
      using FiberField = Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>;

      /// standard constructor
      explicit ActiveStrain(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      /// @{

      /// material id of the passive (elastic) material
      const int matid_passive_;

      /// position (1-based) of the transmembrane voltage in the coupled scalar field
      const int scalid_voltage_;

      /// density
      const double density_;

      /// beta1: dynamics of the voltage-dependent calcium channels
      const double beta1_;

      /// beta2: dynamics of the Ca-ions
      const double beta2_;

      /// voltage threshold above which the contraction is activated
      const double voltage_threshold_;

      /// intensity of the contraction along fiber 1
      const double alpha_fiber1_;

      /// intensity of the contraction along fiber 2 (orthotropic kinematics only)
      const double alpha_fiber2_;

      /// id of a space-time function modulating the contraction along fiber 1 (0 = none)
      const int alpha_funct_fiber1_;

      /// id of a space-time function modulating the contraction along fiber 2 (0 = none)
      const int alpha_funct_fiber2_;

      /// activation source: 0 = coupled scalar field, >0 = id of a space-time voltage function
      const int source_activation_;

      /// kinematics of the active deformation gradient
      const ActiveStrainKinematics kinematics_;

      /// optional fiber 1 direction field (overrides the fiber definition of the element)
      const std::optional<FiberField> fiber1_field_;

      /// optional fiber 2 direction field (overrides the fiber definition of the element)
      const std::optional<FiberField> fiber2_field_;

      /// @}

      /// whether the fiber directions are taken from the material input fields
      [[nodiscard]] bool has_fiber_fields() const { return fiber1_field_.has_value(); }

      /// whether only fiber 1 is used (transversely isotropic active deformation)
      [[nodiscard]] bool is_transversely_isotropic() const
      {
        return kinematics_ == ActiveStrainKinematics::transversely_isotropic;
      }
    };
  }  // namespace PAR

  class ActiveStrainType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "ActiveStrainType"; }

    static ActiveStrainType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ActiveStrainType instance_;
  };

  /*!
   * \brief Active strain material for 3D solids (gastric electromechanics).
   *
   * 3D solid analogue of MAT_Membrane_ActiveStrain. The material wraps an arbitrary passive 3D
   * solid material and applies a multiplicative active strain split of the deformation gradient
   * \f$\mathbf{F} = \mathbf{F}_e \mathbf{F}_a\f$. Two isochoric active deformations are available
   * (ACTIVE_STRAIN_TYPE):
   *
   * - orthotropic (default): two fibers contract and the normal direction compensates the volume,
   *   \f[
   *     \mathbf{F}_a = \mathbf{I} - \gamma_1 \mathbf{f}_1 \otimes \mathbf{f}_1
   *                              - \gamma_2 \mathbf{f}_2 \otimes \mathbf{f}_2
   *                              + \gamma_n \mathbf{n} \otimes \mathbf{n},
   *     \qquad (1 - \gamma_1)(1 - \gamma_2)(1 + \gamma_n) = 1,
   *   \f]
   *   with the orthonormal fiber directions \f$\mathbf{f}_1, \mathbf{f}_2\f$ and
   *   \f$\mathbf{n} = \mathbf{f}_1 \times \mathbf{f}_2\f$.
   * - transversely_isotropic: one fiber contracts and the transverse plane expands isotropically,
   *   \f[
   *     \mathbf{F}_a = (1 - \gamma_1) \mathbf{f}_1 \otimes \mathbf{f}_1
   *                  + (1 - \gamma_1)^{-1/2} (\mathbf{I} - \mathbf{f}_1 \otimes \mathbf{f}_1).
   *   \f]
   *
   * The contractions are \f$\gamma_i = \alpha_i\, m_i(\mathbf{X}, t)\, \gamma\f$ with the
   * intensities ALPHA_FIBER1/2, the optional modulation functions ALPHAFUNCT_FIBER1/2 and the
   * activation
   * \f[
   *   \gamma = \left(1 - e^{-\beta_1 (V - V_{th})}\right) \left(1 - e^{-\beta_2 (V -
   * V_{th})}\right)
   *   \quad \text{for } V > V_{th},
   * \f]
   * which is driven by a (transmembrane) voltage \f$V\f$ that is either taken from a coupled scalar
   * transport field (SOURCE_ACTIVATION 0, requires a solid-scatra element) or prescribed by a
   * space-time function (SOURCE_ACTIVATION > 0, e.g. a traveling activation wave).
   *
   * The passive material is evaluated with the elastic deformation gradient \f$\mathbf{F}_e\f$ and
   * its stress and tangent are pulled back to the reference configuration:
   * \f$\mathbf{S} = \mathbf{F}_a^{-1} \mathbf{S}_e \mathbf{F}_a^{-T}\f$.
   *
   * Fiber directions are taken from the optional material input fields FIBER1 and FIBER2 (constant,
   * element-wise or nodal data, e.g. from a VTU mesh) if given. Otherwise they are read from the
   * element definition, either from the CIR-AXI-RAD coordinate system (fiber 1 = AXI,
   * fiber 2 = CIR, normal = RAD) or from the element fibers FIBER1 and FIBER2. Fiber 1 is
   * normalized, fiber 2 is orthogonalized with respect to fiber 1 and normalized. The transversely
   * isotropic kinematics only use fiber 1.
   */
  class ActiveStrain : public So3Material
  {
   public:
    /// constructor for an empty material object
    ActiveStrain();

    /// constructor for the material given the material parameters
    explicit ActiveStrain(Mat::PAR::ActiveStrain* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return ActiveStrainType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::nonlinearTotLag)
      {
        FOUR_C_THROW(
            "Element and material kinematics are not compatible. Use nonlinear total Lagrangian "
            "kinematics (KINEM nonlinear) in your element definition.");
      }
    }

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_activestrain;
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ActiveStrain>(*this);
    }

    [[nodiscard]] double density() const override { return params_->density_; }

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void post_setup(const Teuchos::ParameterList& params, int eleGID) override;

    void update() override;

    void reset_step() override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    /// orthonormal triad: fiber 1, fiber 2, normal (fiber 2 and normal are unused and zero for
    /// transversely isotropic kinematics)
    using FiberTriad = std::array<Core::LinAlg::Tensor<double, 3>, 3>;

    /// build the triad from the given directions according to the active strain kinematics
    [[nodiscard]] FiberTriad make_fiber_triad(const Core::LinAlg::Tensor<double, 3>& fiber1,
        const std::optional<Core::LinAlg::Tensor<double, 3>>& fiber2) const;

    /// triad at the current evaluation point (from the input fields or the element)
    [[nodiscard]] FiberTriad evaluate_fiber_triad(
        int eleGID, const EvaluationContext<3>& context) const;

    /// (transmembrane) voltage at the current evaluation point
    [[nodiscard]] double evaluate_voltage(
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context) const;

    /// activation parameter gamma from the threshold law
    [[nodiscard]] double evaluate_activation(double voltage) const;

    /// inverse of the (isochoric) active deformation gradient
    [[nodiscard]] Core::LinAlg::Tensor<double, 3, 3> evaluate_inverse_active_deformation_gradient(
        double gamma, const FiberTriad& triad, const EvaluationContext<3>& context) const;

    /// my material parameters
    Mat::PAR::ActiveStrain* params_;

    /// passive (elastic) material
    std::shared_ptr<Mat::So3Material> matpassive_;

    /// triad defined on the element (unused if the fiber input fields are given)
    FiberTriad element_triad_{};

    /// (transmembrane) voltage at every Gauss point (for output)
    std::vector<double> voltage_;

    /// activation parameter gamma at every Gauss point (for output)
    std::vector<double> activation_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
