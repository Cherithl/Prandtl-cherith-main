#pragma once

#include "ConditionFactory.hpp"

namespace Prandtl
{

// Lid-driven Cavity initial condition function
std::function<void(const Vector&, Vector&)> LidDrivenCavityIC(real_t Ma, real_t gamma)
{
    return [Ma, gamma] (const Vector &x, Vector &y)
    {
        real_t p = 1.0 / (Ma * Ma * gamma);
        y(0) = 1.0;
        y(1) = std::sin(x(0)) * std::cos(x(1)) * std::cos(x(2));
        y(2) = -std::cos(x(0)) * std::sin(x(1)) * std::cos(x(2));
        y(3) = p / (gamma - 1.0);
    };
}

// Lid-driven Cavity heat flow boundary condition scalar function for adiabatic walls and lid
std::function<real_t(const Vector&)> LidDrivenCavityAdiaBCFunction()
{
    return [] (const Vector &x)
    {
        return 0.0;
    };
}

// Lid-driven Cavity heat flow boundary condition scalar for adiabatic walls and lid
const BC_Scalar LidDrivenCavityAdiaBCScalar = 0.0;

// Lid-driven Cavity velocity boundary condition function for walls
std::function<void(const Vector&, Vector&)> LidDrivenCavityWallVelBCFunction()
{
    return [] (const Vector &x, Vector &vel)
    {
        vel(0) = 0.0;
        vel(1) = 0.0;
    };
}
// Lid-driven Cavity velocity boundary condition vector for walls
const BC_Vector LidDrivenCavityWallVelBCVector({0.0, 0.0});

// Lid-driven Cavity velocity boundary condition function for lid
std::function<void(const Vector&, Vector&)> LidDrivenCavityLidVelBCFunction(real_t Re, real_t mu)
{
    return [Re, mu] (const Vector &x, Vector &vel)
    {
        vel(0) = Re * mu / 2.0;
        vel(1) = 0.0;
    };
}

// Lid-driven Cavity velocity boundary condition vector for lid
const BC_Vector LidDrivenCavityLidVelBCVector({1.0, 0.0});

// Registration helper that automatically registers these functions
// along with associated boundary marker arrays.
struct RegisterLidDrivenCavity
{
    RegisterLidDrivenCavity()
    {
        // Register initial condition.
        ConditionFactory::Instance().RegisterInitialCondition2("LidDrivenCavityIC", LidDrivenCavityIC);

        // Register boundary conditions with functions.
        ConditionFactory::Instance().RegisterScalarFunctionBoundaryCondition0("LidDrivenCavityAdiaBCFunction", LidDrivenCavityAdiaBCFunction);
        ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition0("LidDrivenCavityWallVelBCFunction", LidDrivenCavityWallVelBCFunction);
        ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition2("LidDrivenCavityLidVelBCFunction", LidDrivenCavityLidVelBCFunction);
        
        // Register boundary conditions with constant scalars/vectors.
        ConditionFactory::Instance().RegisterScalarBoundaryCondition("LidDrivenCavityAdiaBCScalar", LidDrivenCavityAdiaBCScalar);
        ConditionFactory::Instance().RegisterVectorBoundaryCondition("LidDrivenCavityWallVelBCVector", LidDrivenCavityWallVelBCVector);
        ConditionFactory::Instance().RegisterVectorBoundaryCondition("LidDrivenCavityLidVelBCVector", LidDrivenCavityLidVelBCVector);

    }
};
// Global static instance to ensure registration happens at startup.
static RegisterLidDrivenCavity regLidDrivenCavity;

}
