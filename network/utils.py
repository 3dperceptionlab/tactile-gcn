import network.gcn_network

networks = { 'GCN_32_64' : network.gcn_network.GCN_32_64 }

def get_network(name, numFeatures, numClasses):  
  return networks[name](numFeatures, numClasses)