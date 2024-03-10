from torch import optim


def select_optimizer_scheduler(model, args):
    """Documentation for a function.

    More details.
    """
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
            amsgrad=True,
            foreach=True,
        )
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "radam":
        optimizer = optim.RAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "nadam":
        optimizer = optim.NAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            foreach=True,
        )
    elif args.optimizer == "lbfgs":
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=args.learning_rate,
        )
    else:
        raise ValueError("Unknown optimizer")

    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=2500, eta_min=0.1 * args.learning_rate
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError("Unknown scheduler")
    return optimizer, scheduler
