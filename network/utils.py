import network.gcn_network

networks = {
                'GCN_test' : network.gcn_network.GCN_test,
                'GCN_32' : network.gcn_network.GCN_32,
                'GCN_32_64' : network.gcn_network.GCN_32_64,
                'GCN_32_64_128' : network.gcn_network.GCN_32_64_128
}

def get_network(name, numFeatures, numClasses):  
  return networks[name](numFeatures, numClasses)
