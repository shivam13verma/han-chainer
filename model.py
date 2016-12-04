from keras.models import Sequential
from keras.layers import Dense, Activation



model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_shape=(90582, 517)))
model.add(GRU(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(1))
model.add(Activation('softmax'))


#word-gru layer
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(output_dim=128, return_sequences=True))



#word-attention
model = Sequential()
model.add(Dense(50, input_dim=100, init='uniform'))
model.add(Activation('tanh'))

#sentence-gru layer

#sentence-attention

def build_model(opts, verbose=False):
    k = 2 * opts.lstm_units  # 300
    L = opts.xmaxlen  # 20
    N = opts.xmaxlen + opts.ymaxlen + 1  # for delim
    print "x len", L, "total len", N
    print "k", k, "L", L

    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    x = Embedding(output_dim=opts.emb, input_dim=opts.max_features, input_length=N, name='x')(main_input)
    drop_out = Dropout(0.1, name='dropout')(x)
    lstm_fwd = LSTM(opts.lstm_units, return_sequences=True, name='lstm_fwd')(drop_out)
    lstm_bwd = LSTM(opts.lstm_units, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
    bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
    drop_out = Dropout(0.1)(bilstm)
    h_n = Lambda(get_H_n, output_shape=(k,), name="h_n")(drop_out)
    Y = Lambda(get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
    Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
    Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
    WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name="merged", mode='sum')
    M = Activation('tanh', name="M")(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

    r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)

    r = Reshape((k,), name="r")(r_)

    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
    merged = merge([Wr, Wh], mode='sum')
    h_star = Activation('tanh')(merged)
    out = Dense(3, activation='softmax')(h_star)
    output = out
    model = Model(input=[main_input], output=output)
    if verbose:
        model.summary()
    # plot(model, 'model.png')
    # # model.compile(loss={'output':'binary_crossentropy'}, optimizer=Adam())
    # model.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam(options.lr))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(options.lr))
    return model



def compute_acc(X, Y, vocab, model, opts):
    scores = model.predict(X, batch_size=options.batch_size)
    prediction = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        l = np.argmax(scores[i])
        prediction[i][l] = 1.0
    assert np.array_equal(np.ones(prediction.shape[0]), np.sum(prediction, axis=1))
    plabels = np.argmax(prediction, axis=1)
    tlabels = np.argmax(Y, axis=1)
    acc = accuracy(tlabels, plabels)
    return acc, acc
