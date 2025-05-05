#include "mode_handler.h" //in minizero/console/mode_handler.h

int main(int argc, char* argv[])
{
    minizero::console::ModeHandler mode_handler;
    mode_handler.run(argc, argv);
    return 0;
}
