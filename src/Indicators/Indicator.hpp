#pragma once

#include "mfem.hpp"

namespace Prandtl
{

using namespace mfem;

class Indicator
{
protected:
    std::shared_ptr<ParFiniteElementSpace> vfes;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::shared_ptr<ParGridFunction> eta;

    Array<int> vdof_indices, ind_indx;
    Vector el_vdofs, ind_dof;
    int num_equations, ndofs, order;
    Vector state;
public:
    Indicator(std::shared_ptr<ParFiniteElementSpace> vfes,
              std::shared_ptr<ParFiniteElementSpace> fes0,
              std::shared_ptr<ParGridFunction> eta);
    virtual void CheckSmoothness(const Vector &x) = 0;
    virtual ~Indicator() = default;
};
}
