#include "SupersonicInflowBdrFaceIntegrator.hpp"

namespace Prandtl
{

// Constructor for SupersonicInflowBdrFaceIntegrator with a variable (space- and/or time-dependent) conservative state
SupersonicInflowBdrFaceIntegrator::SupersonicInflowBdrFaceIntegrator(
    std::shared_ptr<LiftingScheme> liftingScheme, const NumericalFlux &rsolver, const int Np,
    const real_t &time, real_t gamma, VectorFunctionCoefficient &conserv_state_fun, bool t_dependent)
    : BdrFaceIntegrator(liftingScheme, rsolver, Np, time, gamma, false, t_dependent),
    conserv_state_fun(conserv_state_fun) {}

// Constructor for SupersonicInflowBdrFaceIntegrator with a constant (space- and/or time-dependent) conservative state
SupersonicInflowBdrFaceIntegrator::SupersonicInflowBdrFaceIntegrator(
    std::shared_ptr<LiftingScheme> liftingScheme, const NumericalFlux &rsolver, const int Np,
    const real_t &time, real_t gamma, const Vector &conserv_state)
    : BdrFaceIntegrator(liftingScheme,rsolver, Np, time, gamma, true, false),
    const_state(conserv_state), conserv_state_fun(num_equations, std::function<void(const Vector&, Vector&)>())
{
}

void SupersonicInflowBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    if (constant)
    {
        state2 = const_state;
    }
    else
    {
        if (t_dependent)
        {
            conserv_state_fun.SetTime(time);
        }
        conserv_state_fun.Eval(state2, Tr, ip);
    }
}

void SupersonicInflowBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SupersonicInflowBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SupersonicInflowBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}