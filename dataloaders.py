from ERIC_datasets import MNIST_dataloaders, GTSRB_dataloaders, PASCALanimals_dataloaders, PASCALall_dataloaders, \
    PLACES_dataloaders
def get_dataloader(dataset_name, params, train_shuffle=True):
    """
    Returns the dataloaders for a given dataset
    :param dataset_name: name of the dataset
    :param params: dictionary of parameters
    :return: train_loader, val_loader, test_loader, num_classes, class_weights
    """
    if dataset_name == "MNIST":
        train_loader, val_loader, test_loader, num_classes, class_weights = MNIST_dataloaders(params["batch_size"],train_shuffle=train_shuffle)
    if dataset_name == "GTSRB":
        train_loader, val_loader, test_loader, num_classes, class_weights = GTSRB_dataloaders(params["batch_size"],train_shuffle=train_shuffle)
    if dataset_name == "PLACES2":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom"],train_shuffle=train_shuffle)
    if dataset_name == "PLACES3":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom", "kitchen"],train_shuffle=train_shuffle)
    if dataset_name == "PLACES5":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room"],train_shuffle=train_shuffle)
    if dataset_name == "PLACES10":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office",
                        "waiting_room", "conference_room", "hotel_room"],train_shuffle=train_shuffle)
    if dataset_name == "dedrf":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["desert_road", "driveway", "forest_road"],train_shuffle=train_shuffle)
    if dataset_name == "dedrh":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["desert_road", "driveway", "highway"],train_shuffle=train_shuffle)
    if dataset_name == "dedrs":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["desert_road", "driveway", "street"],train_shuffle=train_shuffle)
    if dataset_name == "defs":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["desert_road", "forest_road", "street"],train_shuffle=train_shuffle)
    if dataset_name == "defh":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["desert_road", "forest_road", "highway"],train_shuffle=train_shuffle)
    if dataset_name == "dehs":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["desert_road", "highway", "street"],train_shuffle=train_shuffle)
    if dataset_name == "drfh":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["driveway", "forest_road", "highway"],train_shuffle=train_shuffle)
    if dataset_name == "drfs":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["driveway", "forest_road", "street"],train_shuffle=train_shuffle)
    if dataset_name == "drhs":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["driveway", "highway", "street"],train_shuffle=train_shuffle)
    if dataset_name == "fhs":
        train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["forest_road", "highway", "street"],train_shuffle=train_shuffle)
    return train_loader, val_loader, test_loader, num_classes, class_weights

def get_class_list(dataset):
    if dataset == "PLACES2":
        class_list = ["bathroom", "bedroom"]
    if dataset == "PLACES3":
        class_list = ["bathroom", "bedroom", "kitchen"]
    if dataset == "PLACES5":
        class_list = ["bathroom", "bedroom", "kitchen", "dining_room", "living_room"]
    if dataset == "PLACES10":
        class_list = ["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office",
                      "waiting_room", "conference_room", "hotel_room"]
    if dataset == "dedrf":
        class_list = ["desert_road", "driveway", "forest_road"]
    if dataset == "dedrh":
        class_list = ["desert_road", "driveway", "highway"]
    if dataset == "dedrs":
        class_list = ["desert_road", "driveway", "street"]
    if dataset == "defs":
        class_list = ["desert_road", "forest_road", "street"]
    if dataset == "defh":
        class_list = ["desert_road", "forest_road", "highway"]
    if dataset == "dehs":
        class_list = ["desert_road", "highway", "street"]
    if dataset == "drfh":
        class_list = ["driveway", "forest_road", "highway"]
    if dataset == "drfs":
        class_list = ["driveway", "forest_road", "street"]
    if dataset == "drhs":
        class_list = ["driveway", "highway", "street"]
    if dataset == "fhs":
        class_list = ["forest_road", "highway", "street"]
    return class_list