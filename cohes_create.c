#include <petscdmplex.h>

typedef struct {
  char      file[PETSC_MAX_PATH_LEN];
  PetscBool interp, dist;
} OptCtx;

PetscErrorCode ProcessOpts(MPI_Comm comm, OptCtx *ctx)
{
  PetscBool      fflag = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ctx->file[0] = '\0';
  ctx->interp = PETSC_TRUE;
  ctx->dist = PETSC_TRUE;
  ierr = PetscOptionsBegin(comm, NULL, "Cohesive Insertion Program Options", NULL);CHKERRQ(ierr);
  {
    ierr = PetscOptionsString("-filename", "Mesh input file", "DMPlexCreateFromFile", ctx->file, ctx->file, PETSC_MAX_PATH_LEN, &fflag); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate mesh", "DMPlexInterpolate", ctx->interp, &ctx->interp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-distribute", "Distribute mesh", "DMPlexDistribute", ctx->dist, &ctx->dist, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!fflag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"User must supply input file");
  PetscFunctionReturn(0);
}

PetscErrorCode InitializeMesh(MPI_Comm comm, OptCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateFromFile(comm, ctx->file, ctx->interp, dm);CHKERRQ(ierr);
  if (ctx->dist) {
    DM dmp;

    ierr = DMPlexDistribute(*dm, 1, NULL, &dmp);CHKERRQ(ierr);
    if (dmp) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmp;
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  OptCtx         ctx;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOpts(comm, &ctx);CHKERRQ(ierr);

  ierr = DMPlexCreateFromFile(comm, ctx.file, ctx.interp, &dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-pre_partition_view");CHKERRQ(ierr);
  if (ctx.dist) {
    DM dmp;

    ierr = DMPlexDistribute(dm, 1, NULL, &dmp);CHKERRQ(ierr);
    if (dmp) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm = dmp;
    }
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
