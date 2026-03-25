#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"

namespace Prandtl
{

// ============================================================================
// EOS: Ideal single-species gas using PhysicsConstants
// ============================================================================
  struct IdealSingleGasEOS
  {
    // ---- helpers on conservative state --------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t R_gas(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return phys.R_gas;
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t density(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
        return S.mass(L); // this is "rho" (mass density)
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t rhoE(const PhysicsConstants &phys, const StateLayout &L,
                       const StateView &S) const
    {
        return S.energy(L);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t momentum_sq(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
        const int dim = L.dim;   // uses state layout
        real_t m2 = 0;
        for (int d = 0; d < dim; ++d)
        {
          const real_t m = S.momentum(L,d);
          m2 += m * m;
        }
        return m2;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                         const StateView &S) const
    {
      // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
      const real_t rho  = density(phys, L, S);
      const real_t m2   = momentum_sq(phys, L, S);
      return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                          const StateView &S) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return rhoE(phys, L, S) - kinetic_energy_density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_from_pressure(const PhysicsConstants &phys, const StateLayout &L,
                                                const StateView &S, real_t pressure) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
      return pressure / (phys.gamma - 1.0);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                           const StateView &S) const
    {
        // e = (rho*e) / rho
      const real_t rho  = density(phys, L, S);
      const real_t rhoe = internal_energy_density(phys, L, S);
      return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t pressure(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S) const
    {
      // p = (gamma - 1) * (rho*E - 0.5*|rho*u|^2 / rho)
      const real_t rhoe = internal_energy_density(phys, L, S);
      return phys.gammaM1 * rhoe;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t gamma(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return phys.gamma;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t temperature(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      // p = rho*R*T  =>  T = p / (rho*R)
      const real_t rho = density(phys, L, S);
      const real_t p   = pressure(phys, L, S);
      return p / (rho * phys.R_gas);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const PhysicsConstants &phys, const StateLayout  &L,
                                 const StateView &S, const real_t *grad_rho,
                                 const real_t *grad_p, real_t *grad_t) const
    {
      for(int i = 0; i < L.dim; i++){
        grad_t[i] = grad_p[i]; // CL NOTE : we store T_xi in contiguous gradient array for LTE (W=[rho, u, v, w , T])
      }  
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      // a^2 = gamma * p / rho
      const real_t rho = density(phys, L, S);
      const real_t p   = pressure(phys, L, S);
      return std::sqrt(phys.gamma * p / rho);
    }
    
    // cp is constant for ideal gas
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t cp(const PhysicsConstants &phys, const StateLayout &L,
                     const StateView & /*S*/) const
    {
        return phys.cp;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t entropy(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
      const real_t p = pressure(phys, L, S);
      const real_t gamma = phys.gamma;
      // TODO: Augment for correct treatment of passive scalars
      return std::log(p) - gamma * std::log(S.mass(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const PhysicsConstants &phys, const StateLayout &L,
                              const InStateView &S, OutStateView &E) const
    {
      const real_t rho = S.mass(L);
      const real_t T = temperature(phys, L, S);
      const real_t v2o2 = kinetic_energy_density(phys, L, S) / rho;
      const real_t beta = 1 / T;

      const real_t ent_1 = 0;

      E.set_mass(L, ent_1);
      int dim = L.dim;
      int num_scalars = L.num_scalars;
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(L, idim, beta * S.velocity(L, idim));
      }
      E.set_energy(L, -beta);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const PhysicsConstants &phys, const StateLayout &L,
                                          const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim) const
    {
      const real_t T = temperature(phys, L, S);

      dPrim.set_mass(L, 0.0); // CL NOTE : won't be using density gradient in LTE (if W = [rho, u, v, w , T])

      int dim = L.dim;
      for(int idim = 0; idim < dim; idim++){
        dPrim.set_momentum(L, idim, dE.momentum(L, idim)*T + T*S.velocity(L, idim)*dE.energy(L));
      }
      dPrim.set_energy(L, T * T * dE.energy(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                     const InStateView &Se, OutStateView &Sc) const
    {
      int dim = L.dim;
      const real_t beta = -Se.energy(L);
      real_t k = 0.0;
      real_t vel[3];
      for(int idim = 0;idim < dim;idim++){
        vel[idim] = Se.momentum(L, idim)/beta;
        k += vel[idim]*vel[idim];
      }
      const real_t gamma = phys.gamma;
      const real_t s = gamma - (Se.mass(L) + 0.5*k*beta)*(gamma - 1.);
      const real_t rho = std::pow(std::exp(-s)/beta, 1.0/(gamma - 1));
      Sc.set_mass(L, rho);
      Sc.set_energy(L, rho*(1.0/(beta*(gamma-1.)) + 0.5*k));
      for(int idim = 0;idim < dim;idim++){
        Sc.set_momentum(L, idim, rho*vel[idim]);
      }
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const PhysicsConstants &phys, const StateLayout &L,
                         const StateView &S, real_t u[3]) const
    {
      const int dim = L.dim;
      for (int d = 0; d < dim; ++d)
        {
          u[d] = S.velocity(L, d);
        }
      for (int d = dim; d < 3; ++d)
        {
          u[d] = real_t(0);
        }
    }
  };
}
