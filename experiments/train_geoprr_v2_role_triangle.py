"""Train the three-role association or its matched marginal-product deletion."""
from experiments.geoprr_v2_role_triangle import RoleTriangleReader, role_triangle_losses
from experiments.train_geoprr_v2_reference_relation import main as train_reference_reader


def main(argv=None):
    return train_reference_reader(argv, reader_class=RoleTriangleReader,
                                  loss_function=role_triangle_losses,
                                  full_initialization="full_best48")


if __name__ == "__main__":
    raise SystemExit(main())
