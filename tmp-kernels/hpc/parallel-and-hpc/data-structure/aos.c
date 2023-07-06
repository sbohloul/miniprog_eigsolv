#include <stdio.h>
#include <math.h>
// #include <stdlib.h>

struct point
{
    double x, y, z;
};

int main()
{
    const int ncell = 10;
    struct point cell[ncell];
    double radius[ncell];

    for (int i = 0; i < ncell; i++)
    {
        radius[i] = cell[i].x * cell[i].x;
        radius[i] = cell[i].y * cell[i].y + radius[i];
        radius[i] = cell[i].z * cell[i].z + radius[i];
        // radius[i] = sqrt(radius[i]);
        // radius[i] = sqrt(cell[i].x * cell[i].x + cell[i].y * cell[i].y + cell[i].z * cell[i].z);
    }

    for (int i = 0; i < ncell; i++)
    {
        printf("%f\n", cell[i].x * cell[i].x);
    }
    return 0;
}
