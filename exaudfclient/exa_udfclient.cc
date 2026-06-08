#include "exa_common/exa_udf_clients.h"

int main(int argc, char **argv) {
    ExaUdfClient client;
    return client.startClientBase(argc, argv);
}