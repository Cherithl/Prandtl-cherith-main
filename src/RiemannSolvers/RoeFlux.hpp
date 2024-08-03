#pragma once

#include "mfem.hpp"

namespace Prandtl
{

using namespace mfem;

/**
 * @brief Roe flux,
 * F̂ n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
 * where λ is the maximum characteristic velocity
 *
 */
class RoeFlux : public RiemannSolver
{
public:
   RoeFlux(const FluxFunction &fluxFunction)
      : RiemannSolver(fluxFunction)
   {
#ifndef MFEM_THREAD_SAFE
      fluxN1.SetSize(fluxFunction.num_equations);
      fluxN2.SetSize(fluxFunction.num_equations);
#endif
   }

   /**
    * @brief  hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    */
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override;

protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1, fluxN2;
#endif
};

}