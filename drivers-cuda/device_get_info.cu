#include <iostream>

using namespace std;

int main(int argc, char **argv)
{

    cout << "==========================" << endl;
    cout << "Getting GPU(s) information" << endl;
    cout << "==========================" << endl;

    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cout << "driverVersion = " << driverVersion << endl;

    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cout << "runtimeVersion = " << runtimeVersion << endl;

    int numDevices = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&numDevices);

    if (cudaStatus != cudaSuccess)
    {
        cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(cudaStatus) << endl;
    }
    cout << "numDevices: " << numDevices << endl;
    cout << endl;

    for (int i = 0; i < numDevices; i++)
    {
        cudaDeviceProp prop;
        cudaStatus = cudaGetDeviceProperties(&prop, i);

        cout << "Device " << i << ":" << endl;
        cout << "prop.name " << prop.name << endl;
        cout << "prop.major " << prop.major << endl;
        cout << "prop.minor " << prop.minor << endl;
        cout << "prop.multiProcessorCount " << prop.multiProcessorCount << endl;
        cout << "prop.maxThreadsPerMultiProcessor " << prop.maxThreadsPerMultiProcessor << endl;
        cout << "prop.clockRate  " << prop.clockRate << endl;
        cout << "prop.totalGlobalMem " << prop.totalGlobalMem << endl;
        cout << endl;
    }

    return 0;
}