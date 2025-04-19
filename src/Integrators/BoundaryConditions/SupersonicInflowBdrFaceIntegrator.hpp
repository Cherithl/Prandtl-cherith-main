# pragma once

#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{
using namespace mfem;

class SupersonicInflowBdrFaceIntegrator : public BdrFaceIntegrator
{
private:
    Vector const_state;
    VectorFunctionCoefficient conserv_state_fun;
public:
    SupersonicInflowBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, const NumericalFlux &rsolver, const int Np, const real_t &time, real_t gamma, VectorFunctionCoefficient &conserv_state_fun, bool t_dependent = false);
    SupersonicInflowBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, const NumericalFlux &rsolver, const int Np, const real_t &time, real_t gamma, const Vector &conserv_state);
    
    virtual void ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;

    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override; 
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override; 
};

}