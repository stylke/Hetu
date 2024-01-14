# Multi Distributed Strategy Support:

## C++端修改：

multi ds support: graph设置STRATEGY_NUM和CUR_STRATEGY_ID来支持多组distributed states + device group的同时推导，其中STRATEGY_NUM为总共的策略数量，CUR_STRATEGY_ID为当前使用的策略。首先入度为0的parallel placeholder op和parallel variable op都需要支持读入多组distributed states、device group、provided data等，这里对tensor的distributed states和op_meta的device group以及variable op的provided data等接口都做了修改，因为多组策略，必然对应着多组distributed states、多组device group，以及parallel variable op的多组provided data，这里只需要设置特定的CUR_STRATEGY_ID，那么用和原来一致的接口获取的就是当前策略下的对应属性，最大程度的兼容了原始接口；

- 具体来说，只有在parallel placeholder op, parallel varibale op的初始化、op def创建时调用op的deduce states、comm op的do gradient、graph.cc在optimizer.minimize中生成backward的comm op，这几种case下，需要循环遍历CUR_STRATEGY_ID=[0: STRATEGY_NUM)来对多组ds进行分别推导和赋值，基本框架如下：

  ~~~bash
  for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
  	graph.CUR_STRATEGY_ID = cur_strategy_id;
  	// do deduce states
  	output->set_distributed_states(ds);
  }
  graph.CUR_STRATEGY_ID = 0;
  ~~~

- comm op为了适应multi ds，其接口从原先的comm(input, ds) -> comm(input, multi_ds)，tensor、op_meta均有不同程度的修改，详情见代码

综上，define and run graph只需要确定好想要哪种策略，并设置对应的exec graph的CUR_STRATEGY_ID就可以直接切换至该策略，其他接口调用都无需做任何改动，非常方便。注意：

- 我们只在define and run做大部分op的deduce states，exec graph的op的ds都是直接copy define and run的，除了那些新建的用于表示p2p的comm op。
- 为了保证define and run graph用多组ds推导出来的计算图一致，这里只要两个tensor的两组multi ds之间有一组需要通信，那就需要创建一个新的comm op(input, multi_dst_ds)，在具体实例化executable graph的时候，如果选择具体的CUR_STRATEGY_ID后，某些comm op并不需要通信（src_ds=dst_ds），那么就会把它给remove掉。这里有一个小技巧，之前exec graph用comm op的src ds == dst ds来判断是否为P2P_OP，但实际上p2p op只有在exec graph的instantiate阶段才会被动创建，因此但凡是define and run graph中生成的comm op在exec graph的第一轮topo中（手动插入的p2p并不会在第一轮topo中出现，但会在第二轮updated_topo中出现）被判断为是P2P_OP类型，那就是unused comm op，直接remove即可，当然，remove之后会有一些连锁反应需要在instantiate中顺带处理，细节略...

## Python端修改：

对c++端修改的各种适配，具体细节看代码

为了让不同策略能够共享同一计算图，这里c++端重写了EmbeddingLookUp op和VocabParallelCrossEntropy op的逻辑，从而能够兼容不同的分布式策略（主要是tp vocab parallel和pure dp的兼容），python端只需要调用同一个接口即可

提供了能够推导multi ds的python端e2e脚本，GPT2.7B上测试同时推导dp2tp2pp2、dp8、dp4tp2、tp8，经过漫长地debug之后，代码均能正常运行

- examples/nlp/gpt/scripts/train_hetu_gpt_multi_ds_parallel.sh
- examples/nlp/gpt/train_hetu_gpt_multi_ds_parallel.py
- examples/nlp/gpt/hetu_gpt_multi_ds_parallel.py
- python_refactor/hetu/nn/modules/parallel_multi_ds.py

## others

注：实际上shape的推导也可以像distributed states/device group那样推导多组，这里我默认按照第一组策略来推导shape了，因此通过CUR_STRATEGY_ID切换策略后，distributed states、device group、provided data都是正确的，但shape还是第一组的shape，不过要改成多组shape推导应该也挺快的