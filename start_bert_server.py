from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer

model_path = 'D:\\Projects\\stock_predict\\bert\\uncased_L-12_H-768_A-12'

args = get_args_parser().parse_args(['-model_dir', model_path,
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu'])
server = BertServer(args)
server.start()