#include "exa_udfclient_binary_common/exa_udf_client.h"

int main(int argc, char **argv) {
    ExaUdfClient client;
    return client.startClientBase(argc, argv);
}