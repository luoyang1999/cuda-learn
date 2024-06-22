#include<math.h>
#include<stdlib.h>
#include<stdio.h>

const double EPSILON = 1.0e-10;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;


void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main() {
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double *)malloc(M);
    double *y = (double *)malloc(M);
    double *z = (double *)malloc(M);
    for (int i = 0; i < N; i++) {
        x[i] = a;
        y[i] = b;
        z[i] = c;
    }

    add(x, y, z, N);
    check(z, N);
}

void add(const double *x, const double *y, double *z, const int N) {
    for (int i = 0; i < N; i++) {
        z[i] = x[i] + y[i];
    }
}

void check(const double *z, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; i++) {
        if (fabs(z[i] - (a + b)) > EPSILON) {
            printf("Error: z[%d] = %f\n", i, z[i]);
            has_error = true;
            break;
        }
    }
    printf("%s\n", has_error ? "Error" : "Correct");
}