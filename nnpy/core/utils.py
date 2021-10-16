def create_graph(x,layers):
    idx = 0
    g = []
    while True:
        layer = layers[idx]
        if layer.x.data == x.data:
            g.append(layer)
            x = layer.out
            del layers[idx]
            idx = 0
        else:
            if idx == len(layers):
                idx = 0
            else:
                idx+=1

        if len(layers) == 0:
            break

    return g