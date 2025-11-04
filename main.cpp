#include "Simulation.hpp"

using namespace mfem;

int main(int argc, char* argv[])
{
    std::string config_file_path;

    for (int i = 1; i < argc; i++)
    {
        // check if the argument is the -c flag
        if (std::string(argv[i]) == "-c" && i + 1 < argc)
        {
            config_file_path = argv[i + 1];
            break;
        }
    }

    if (config_file_path.empty())
    {
        std::cerr << "\nError: Configuration file not specified." << std::endl;
        std::cerr << "Usage: " << argv[0] << " -c <path/to/config.json>\n" << std::endl;
        return 1;
    }

    Prandtl::Simulation &sim = Prandtl::Simulation::SimulationCreate();
    sim.LoadConfig(config_file_path);
    sim.Run();

}
