#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <functional>
#include <string>
#include "mfem.hpp"

#include "DGSEMOperator.hpp"

#include "DGSEMIntegrator.hpp"
#include "BasicOperations.hpp"
#include "ModalBasis.hpp"
#include "EntropyFilter.hpp"
#include "BdrFaceIntegrator.hpp"
#include "SlipWallBdrFaceIntegrator.hpp"
#include "NoSlipAdiabWallBdrFaceIntegrator.hpp"
#include "NoSlipIsothWallBdrFaceIntegrator.hpp"
#include "SubsonicInflowRVBdrFaceIntegrator.hpp"
#include "SubsonicInflowPtTtAngBdrFaceIntegrator.hpp"
#include "SubsonicOutflowPBdrFaceIntegrator.hpp"
#include "SpecifiedStateBdrFaceIntegrator.hpp"
#include "SupersonicInflowBdrFaceIntegrator.hpp"
#include "SupersonicOutflowBdrFaceIntegrator.hpp"
#include "RiemannInvariantBdrFaceIntegrator.hpp"
#include "LaxFriedrichsFlux.hpp"
#include "ChandrashekarFlux.hpp"
#include "Physics.hpp"
#include "Limiter.hpp"
#include "PerssonPeraireIndicator.hpp"
#include "Simulation.hpp"
#include "NumericalFlux.hpp"
#include "NavierStokesFlux.hpp"
