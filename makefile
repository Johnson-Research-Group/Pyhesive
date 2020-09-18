ALL: cohes_create

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
CURRENT    = $(shell pwd)
NPROCS     = $(shell )

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

cohes_create: cohes_create.o
	${CLINKER} -w -o cohes_create cohes_create.o ${PETSC_LIB}
