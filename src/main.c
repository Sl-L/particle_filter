#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <stdio.h>
#include "arm_math.h"
#include "include/particle_filter/particle_filter.h"

int main(int argc, char **argv)
{
  test_particle_filter();
  return 0;
}