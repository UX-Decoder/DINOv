import os
import json, time
import torch
import torch.distributed as dist


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

def all_gather(x):
    if get_world_size() > 1:
        all_x = [torch.zeros_like(x) for _ in range(get_world_size())]
        torch.distributed.all_gather(all_x, x.detach())
        all_x[torch.distributed.get_rank()] = x
        x = torch.stack(all_x, dim=0)
    return x

def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and 
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])

        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        if os.environ.get('HAND_DEFINE_DIST_URL', 0) == '1':
            pass
        else:
            import util.hostlist as uh
            nodenames = uh.parse_nodelist(os.environ['SLURM_JOB_NODELIST'])
            gpu_ids = [int(node[3:]) for node in nodenames]
            fixid = int(os.environ.get('FIX_DISTRIBUTED_PORT_NUMBER', 0))
            # fixid += random.randint(0, 300)
            port = str(3137 + int(min(gpu_ids)) + fixid)
            args.dist_url = "tcp://{ip}:{port}".format(ip=uh.nodename_to_ip(nodenames[0]), port=port)

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))


    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        world_size=args.world_size, 
        rank=args.rank,
        init_method=args.dist_url,
    )

    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")