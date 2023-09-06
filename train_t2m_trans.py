import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
import torch.nn.functional as F
import torch.nn as nn
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 310)  # 예시로 hidden layer 크기는 512로 설정 
        # humanML -> 310이랑 매치 하지 않을 수 있음 (수정 필요)
        #self.fc2 = nn.Linear(513, output_size)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        #x = self.fc1(x)
        return x
    
mlp_model = MLP(513).cuda()

##### ---- get code ---- #####
for batch in train_loader_token:
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()
    np.save(pjoin(args.vq_dir, name[0] +'.npy'), target)

vocabulary_dict={'swim': 0, 'convulse': 1, 'drank': 2, 'lunge': 3, 'counterclockwise': 4, 'weigh': 5, 'begin': 6, 'land': 7, 'descend': 8, 'gain': 9, 'carry': 10, 'look': 11, 'dribble': 12, 'stabilize': 13, 'define': 14, 'stroll': 15, 'drag': 16, 'stomach': 17, 'remain': 18, 'balance': 19, 'sink': 20, 'side': 21, 'hike': 22, 'ist': 23, 'speed': 24, 'come': 25, 'box': 26, 'stir': 27, 'hang': 28, 'wake': 
29, 'tumble': 30, 'catch': 31, 'clean': 32, 'help': 33, 'flutter': 34, 'predefine': 35, 'swinge': 36, 'swing': 37, 'show': 38, 'can': 39, 'feel': 40, 'open': 41, 'stretch': 42, 'regain': 43, 'shove': 44, 'zoom': 45, 'fight': 46, 'spread': 47, 'defeat': 48, 'balk': 49, 'stumble': 50, 'return': 51, 'dig': 52, 'flex': 53, 'kick': 54, 'block': 55, 'steer': 56, 'rebalance': 57, 'lurch': 58, 'knee': 59, 'inspect': 60, 'change': 61, 'drink': 62, 'wait': 63, 'wave': 64, 'collapse': 65, 'raise': 66, 'stetche': 67, 'bang': 68, 'wipe': 69, 'drum': 70, 'shoot': 71, 'avoid': 72, 'hit': 73, 'retreat': 74, 'strike': 75, 'lurk': 76, 'protect': 77, 'overstep': 78, 'step': 79, 'point': 80, 'shake': 81, 'imitate': 82, 'loosen': 83, 'wawe': 84, 'absorb': 85, 'kneel': 86, 'start': 87, 'support': 88, 'move': 89, 'happen': 90, 'strumble': 91, 'play': 92, 'correct': 93, 'toss': 94, 'visit': 95, 'lefthande': 96, 'accelerate': 97, 'bounce': 98, 'rest': 99, 'seem': 100, 'paddle': 101, 'energize': 102, 'lose': 103, 'hobble': 104, 'punch': 105, 'shrug': 106, 'outstreche': 107, 'leftside': 108, 'sit': 109, 'miss': 110, 'compensate': 111, 'sway': 112, 'wald': 113, 'shout': 114, 'stopd': 115, 'curve': 116, 'wag': 117, 'waltz': 118, 'fast': 119, 'give': 120, 'dance': 121, 'drift': 122, 'stove': 123, 'sideward': 124, 'need': 125, 'omethe': 126, 'fold': 127, 'maintain': 128, 'tilt': 129, 'serve': 130, 'trip': 131, 'tae': 132, 'pivot': 133, 'greet': 134, 'stumbel': 135, 'turn': 136, 'go': 137, 'grip': 138, 'roll': 139, 'appear': 140, 'run': 141, 'balace': 142, 'pat': 143, 'react': 144, 'face': 145, 'interact': 146, 'stare': 147, 'throw': 148, 'laugh': 149, 'clap': 150, 'ture': 151, 'slip': 152, 'shuffle': 153, 'object': 154, 'call': 155, 'alcoholize': 156, 'dodge': 157, 'push': 158, 'pass': 159, 'perturbation': 160, 'againgst': 161, 'assume': 162, 'streche': 163, 'golf': 164, 'sprint': 165, 'whip': 166, 'stop': 167, 'avaoid': 168, 'stroke': 169, 'alternate': 170, 'fall': 171, 'decelerate': 172, 'skateboard': 173, 'extend': 174, 'strirre': 175, 'dangle': 176, 'complete': 177, 'describe': 178, 'beat': 179, 'hold': 180, 'keep': 181, 'put': 182, 'exercise': 183, 'rub': 184, 'flap': 185, 'chest': 186, 'continue': 187, 'wiggle': 188, 'perform': 189, 'walk': 190, 'lower': 191, 'bhind': 192, 'shave': 193, 'squat': 194, 'wawve': 195, 'slap': 196, 'resemble': 197, 'bring': 198, 'crouch': 199, 'jogg': 200, 'counterbalance': 201, 'suggest': 202, 'scream': 203, 'cast': 204, 'crawl': 205, 'beee': 206, 'pour': 207, 'evade': 208, 'prop': 209, 'rise': 210, 'sideware': 211, 'end': 212, 'set': 213, 'bore': 214, 'righthande': 215, 'may': 216, 'watch': 217, 'trurn': 218, 'place': 219, 'person': 220, 'sand': 221, 'mirror': 222, 'pick': 223, 'clockwise': 224, 'search': 225, 'brakedance': 226, 'dwadle': 227, 'shovel': 228, 'eat': 229, 'lay': 230, 'prepare': 231, 'lean': 232, 'kneee': 233, 'rightside': 234, 'try': 235, 'find': 236, 
'puche': 237, 'follow': 238, 'outstretche': 239, 'spin': 240, 'close': 241, 'recover': 242, 'tip': 243, 'form': 244, 'take': 245, 'salute': 246, 'proceed': 247, 'cartwheel': 248, 'bow': 249, 'stand': 250, 'write': 251, 'jog': 252, 'wippe': 253, 'have': 254, 'stomp': 255, 'strum': 256, 'duck': 257, 'rotate': 258, 'mimic': 259, 'sneak': 260, 'perfom': 261, 'tour': 262, 'yanks': 263, 'bump': 264, 'get': 265, 'forward': 266, 'stamp': 267, 'draw': 268, 'sidestep': 269, 'lead': 270, 'disappear': 271, 'let': 272, 'sidekick': 273, 'smash': 274, 'impersonate': 275, 'unk': 276, 'obove': 277, 'want': 278, 'stay': 279, 'bend': 280, 'receive': 281, 'wish': 282, 'resume': 283, 'forwards': 284, 'back': 285, 'lift': 286, 'cross': 287, 'slow': 288, 'indicate': 289, 'climb': 290, 'left': 291, 'touch': 292, 'hand': 293, 'wlak': 294, 'lie': 295, 'hop': 296, 'jump': 297, 'use': 298, 'practice': 299, 'do': 300, 'know': 301, 'perfome': 302, 'continuos': 303, 'tap': 304, 'hurry': 305, 'reach': 306, 'saw': 307, 'make': 308, 'knock': 309}
train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

word_set = set()        
##### ---- Training ---- #####
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)
while nb_iter <= args.total_iter:
    
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, verbs = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 26)
    target = target.cuda()
    label_data = []

    for sentence in verbs:
        sentence_indices = []
        for word in sentence.split():
            if word in vocabulary_dict:
                sentence_indices.append(vocabulary_dict[word])
        label_data.append(sentence_indices)
    
    label_onehot = []
    num_labels=310

    for labels in label_data:
        onehot = np.zeros(num_labels)
        onehot[labels] = 1
        label_onehot.append(onehot)
    label_onehot = np.array(label_onehot)
    label_tensor = torch.tensor(label_onehot).float().cuda()
    #print(label_tensor)
    #for words in verbs:
    #    for word in words.split():  # 공백을 기준으로 단어 분리
    #        word_set.add(word)
    #print(len(word_set),'len(word_set)')
    #print(word_set,'verbs')
    #print(len(target),'len(target)')
    #print(bs,'bs')
    #print(len(clip_text),'len_verbs')
    #print(len(verbs),'len_verbs')

    text = clip.tokenize(clip_text).cuda()
    
    feat_clip_text = clip_model.encode_text(text).float()

    input_index = target[:,:-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                         device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                         device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text)
    final_feature = cls_pred[:,-1,:]
    final_logit = mlp_model(final_feature).cuda()
    #print(label_tensor.shape,'label_tensor.shape')
    #print(final_logit.shape,'final_logit.shape')
    
    #print(final_logit.shape,'final_logit.shape')
    #print(cls_pred.shape,'cls_pred.shape') #(30,51,513)   

    cls_pred = cls_pred.contiguous()

    loss_cls = 0.0
    for i in range(bs):
        # loss function     (26), (26, 513)
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs

        # Accuracy
        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)

        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()
    
    multilabel_loss = F.binary_cross_entropy(final_logit, label_tensor)
    
    loss_cls = loss_cls + multilabel_loss
    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper)

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            