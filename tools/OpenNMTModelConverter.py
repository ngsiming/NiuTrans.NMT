'''
Convert a opennmt-py checkpoint to a NiuTrans.NMT model.
Usage: python3 ModelConverter.py -src <opennmt_models> -tgt <niutrans_nmt_model>
Help: python3 ModelConverter.py -h
Requirements: OpenNMT-py 
'''

import torch
import argparse
import numpy as np
from glob import glob
from struct import pack

parser = argparse.ArgumentParser(
    description='The model converter for NiuTrans.NMT')
parser.add_argument('-src', help='The pattern used to find opennmt checkpoints, e.g., \'checkpoint*\'',
                    type=str, default='checkpoint')
parser.add_argument('-tgt', help='The file name prefix for Niutrans.NMT models',
                    type=str, default='model')
parser.add_argument('-mode', help='Storage mode, FP32 (Default) or FP16', type=str, default='fp32')
args = parser.parse_args()
args.mode = args.mode.lower()



def get_model_parameters(m):
    '''
    get flattend transformer model parameters
    '''
    p = []
    encoder_emb = None
    decoder_emb = None
    decoder_output_w = None
    
    enc_layer_num = m['opt'].enc_layers
    dec_layer_num = m['opt'].dec_layers
    
    # encoder embedding
    encoder_emb = m['model']['encoder.embeddings.make_embedding.emb_luts.0.weight']
    encoder_pos_emb = m['model']['encoder.embeddings.make_embedding.pe.pe']

    enc_prefix = 'encoder.transformer'
    enc_layer = [
        'self_attn.linear_query.weight',
        'self_attn.linear_keys.weight',
        'self_attn.linear_values.weight',
        'self_attn.linear_query.bias',
        'self_attn.linear_keys.bias',
        'self_attn.linear_values.bias',
        'self_attn.final_linear.weight',
        'self_attn.final_linear.bias',
        'feed_forward.w_1.weight',
        'feed_forward.w_1.bias',
        'feed_forward.w_2.weight',
        'feed_forward.w_2.bias',
        'layer_norm.a_2',
        'layer_norm.b_2',
        'feed_forward.layer_norm.a_2',
        'feed_forward.layer_norm.b_2'
    ]
    for i in range(enc_layer_num):
        for e in enc_layer:
            p.append(m['model'][enc_prefix + '.' + str(i) + '.' + e])

    # encoder layer normalization weight & bias
    p.append(m['model']['encoder.layer_norm.a_2'])
    p.append(m['model']['encoder.layer_norm.b_2'])

    # decoder embedding
    try:
        decoder_emb = m['model']['decoder.embeddings.make_embedding.emb_luts.0.weight']
        decoder_pos_emb = m['model']['decoder.embeddings.make_embedding.pe.pe']
    except:
        pass

    dec_prefix = 'decoder.transformer_layers'
    dec_layer = [
        'self_attn.linear_query.weight',
        'self_attn.linear_keys.weight',
        'self_attn.linear_values.weight',
        'self_attn.linear_query.bias',
        'self_attn.linear_keys.bias',
        'self_attn.linear_values.bias',
        'self_attn.final_linear.weight',
        'self_attn.final_linear.bias',
        'layer_norm_1.a_2',
        'layer_norm_1.b_2',
        'context_attn.linear_query.weight',
        'context_attn.linear_keys.weight',
        'context_attn.linear_values.weight',
        'context_attn.linear_query.bias',
        'context_attn.linear_keys.bias',
        'context_attn.linear_values.bias',
        'context_attn.final_linear.weight',
        'context_attn.final_linear.bias',
        'layer_norm_2.a_2',
        'layer_norm_2.b_2',
        'feed_forward.w_1.weight',
        'feed_forward.w_1.bias',
        'feed_forward.w_2.weight',
        'feed_forward.w_2.bias',
        
        'feed_forward.layer_norm.a_2',
        'feed_forward.layer_norm.b_2'
    ]
    for i in range(dec_layer_num):
        for d in dec_layer:
            p.append(m['model'][dec_prefix + '.' + str(i) + '.' + d])

    # decoder layer normalization weight & bias
    p.append(m['model']['decoder.layer_norm.a_2'])
    p.append(m['model']['decoder.layer_norm.b_2'])

    # encoder embedding weight
    p.append(encoder_emb)

    # decoder embedding weight
    if decoder_emb is not None:
        p.append(decoder_emb)
    else:
        print('Sharing encoder decoder embeddings')

    # decoder output weight
    if decoder_output_w is not None:
        p.append(decoder_output_w)
    else:
        print('Sharing decoder input output embeddings')

    return p


with torch.no_grad():

    model_files = glob(args.src)

    for index, model_file in enumerate(model_files):

        print('-' * 120)
        print("source model: \'{}\' ({}/{})".format(model_file, index+1, len(model_files)))
        print("target model: \'{}\'".format(args.tgt + "." + str(index)))
        model = torch.load(model_file, map_location='cpu')

        meta_info = {
            'src_vocab_size': model['model']['encoder.embeddings.make_embedding.emb_luts.0.weight'].shape[0],
            'tgt_vocab_size': model['model']['decoder.embeddings.make_embedding.emb_luts.0.weight'].shape[0],
            'encoder_layer': model['opt'].enc_layers,
            'decoder_layer': model['opt'].dec_layers,
            'ffn_hidden_size': model['model']['encoder.transformer.0.feed_forward.w_1.bias'].shape[0],
            'hidden_size': model['model']['encoder.transformer.0.self_attn.final_linear.bias'].shape[0],
            'emb_size': model['opt'].src_word_vec_size,
            'head_num': 8,
            'max_relative_length': -1,
            'share_all_embeddings': model['opt'].share_embeddings,
            'share_decoder_input_output_embed': 1,
            'max_source_positions': 5000,
        }

        params = get_model_parameters(model)

        print('total params: ', len(params))
        print('total params size: ', sum([p.numel() for p in params]))

        model = model['model']
        with open(args.tgt + "." + str(index) + "." +"name.txt", "w") as name_list:
            for p in model:
                name_list.write("{}\t{}\n".format(p, model[p].shape))

        meta_info_list = [
            meta_info['encoder_layer'],
            meta_info['decoder_layer'],
            meta_info['ffn_hidden_size'],
            meta_info['hidden_size'],
            meta_info['emb_size'],
            meta_info['src_vocab_size'],
            meta_info['tgt_vocab_size'],
            meta_info['head_num'],
            meta_info['max_relative_length'],
            meta_info['share_all_embeddings'],
            meta_info['share_decoder_input_output_embed'],
            meta_info['max_source_positions'],
        ]
        print(meta_info)
        meta_info_list = [int(p) for p in meta_info_list]
        meta_info = pack("i" * len(meta_info_list), *meta_info_list)

        with open(args.tgt + "." + str(index), 'wb') as tgt:
            # part 1: meta info
            tgt.write(meta_info)
                
            # part 2: values of parameters (in FP32 or FP16)
            for p in params:
                if args.mode == 'fp32':
                    values = pack("f" * p.numel(), *
                                (p.contiguous().view(-1).cpu().numpy()))
                    tgt.write(values)
                elif args.mode == 'fp16':
                    values = pack(
                        "e" * p.numel(), *(p.contiguous().view(-1).cpu().numpy().astype(np.float16)))
                    tgt.write(values)
