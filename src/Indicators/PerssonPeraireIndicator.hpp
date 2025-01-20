#pragma once

#include "Indicator.hpp"
#include "ModalBasis.hpp"

namespace Prandtl
{

class PerssonPeraireIndicator : public Indicator
{
private:
    std::shared_ptr<ModalBasis> modalBasis;
    Vector rho_p, modes, modesM1, modesM2;
    Array2D<int> ubdegs;
    Array<int> ubdegs_row;

public:
    PerssonPeraireIndicator(std::shared_ptr<ParFiniteElementSpace> vfes,
                            std::shared_ptr<ParFiniteElementSpace> fes0,
                            std::shared_ptr<ParGridFunction> eta,
                            std::shared_ptr<ModalBasis> modalBasis);
    virtual void CheckSmoothness(const Vector &x) override;
};

}