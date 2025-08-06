#include <iostream>
#include <ostream>

#include "common/log.h"
#include "windows/main_window.h"

int main() {
  autoalg::MainWindow window(800, 800);
  const int result = window.Process();
  DEBUG(MAIN) << "Processed [" << result << "] results";
  return 0;
}
