FROM linux_container

RUN mkdir /exasol_src /exasol_src/build /exasol

COPY src/ /exasol_src/

RUN /exasol_src/build_client.sh /exasol_src /exasol_src/build /exasol
