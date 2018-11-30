import network.gcn_network
import network.chebconv_network

networks = {
                'GCN_test' : network.gcn_network.GCN_test,
                'GCN_32' : network.gcn_network.GCN_32,
                'GCN_32_64' : network.gcn_network.GCN_32_64,
                'GCN_32_64_128' : network.gcn_network.GCN_32_64_128,
                'GCN_8': network.gcn_network.GCN_8,
                'GCN_8_8': network.gcn_network.GCN_8_8,
                'GCN_8_8_16': network.gcn_network.GCN_8_8_16,
                'GCN_8_8_16_16': network.gcn_network.GCN_8_8_16_16,
                'GCN_8_8_16_16_32': network.gcn_network.GCN_8_8_16_16_32,
                'GCN_8_8_16_16_32_32' : network.gcn_network.GCN_8_8_16_16_32_32,
                'GCN_8_8_16_16_32_32_48' : network.gcn_network.GCN_8_8_16_16_32_32_48,
                'GCN_8_8_16_16_32_32_48_48' : network.gcn_network.GCN_8_8_16_16_32_32_48_48,
                'GCN_8_8_16_16_32_32_48_48_64' : network.gcn_network.GCN_8_8_16_16_32_32_48_48_64,
                'GCN_8_8_16_16_32_32_48_48_64_64' : network.gcn_network.GCN_8_8_16_16_32_32_48_48_64_64,
                'GCN_8bn_8bn_16bn_16bn_32bn' : network.gcn_network.GCN_8bn_8bn_16bn_16bn_32bn,
                'GCN_4_4_8_8_16_16_32' : network.gcn_network.GCN_4_4_8_8_16_16_32,

                'ChebConv_8_8' : network.chebconv_network.ChebConv_test,
                'ChebConv_8_16_32' : network.chebconv_network.ChebConv_8_16_32
}

def get_network(name, numFeatures, numClasses):  
  return networks[name](numFeatures, numClasses)
