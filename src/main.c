#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <stdio.h>
#include "arm_math.h"

int main(int argc, char **argv)
{
  arm_matrix_instance_f32 A;
  float32_t data[4] = {1, 2, 3, 4};
  arm_mat_init_f32(&A, 2, 2, data);

  printf("Matrix initialized: [%f %f; %f %f]\n",
         data[0], data[1], data[2], data[3]);

  return 0;
}