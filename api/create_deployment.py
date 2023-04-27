from argparse import ArgumentParser

from mlflow.deployments import get_deploy_client


def create_deployment(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    config = {
        "MODEL_FILE": parser_args["model_file"],
        "HANDLER": parser_args["handler"],
    }

    if parser_args["export_path"] != "":
        config["EXPORT_PATH"] = parser_args["export_path"]

    result = plugin.create_deployment(
        name=parser_args["deployment_name"],
        model_uri=parser_args["model_uri"],
        config=config,
    )

    print("Deployment {result} created successfully".format(result=result["name"]))


if __name__ == "__main__":
    parser = ArgumentParser(description="test")

    parser.add_argument(
        "--target",
        type=str,
        default="torchserve",
        help="MLflow target (default: torchserve)",
    )

    parser.add_argument(
        "--deployment_name",
        type=str,
        default="cifar_test",
        help="Deployment name (default: cifar_test)",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        default="api/models.py",
        help="Model file path (default: api/models.py)",
    )

    parser.add_argument(
        "--handler",
        type=str,
        default="api/models.py",
        help="Handler file path (default: api/models.py)",
    )


    parser.add_argument(
        "--model_uri",
        type=str,
        default="resnet.pth",
        help="List of extra files",
    )

    parser.add_argument(
        "--export_path",
        type=str,
        default="model_store",
        help="Path to model store (default: 'model_store')",
    )

    args = parser.parse_args()

    create_deployment(vars(args))