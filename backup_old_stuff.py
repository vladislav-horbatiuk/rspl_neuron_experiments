def recurrent_cells_forward(cells, inp, contexts):
    for i, cell in enumerate(cells):
        inp = cell(inp, contexts[i])
        contexts[i] = inp
    return inp


def get_pos_emb(ts, freqs):
    return torch.tensor([np.sin(f * ts) for f in freqs]).to(DEVICE)


class Sequence(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_frequencies=()):
        super(Sequence, self).__init__()
        recurrent_cells = []
        self.embedding_frequencies = embedding_frequencies
        input_size = input_size + len(embedding_frequencies)
        for i in range(NUM_CELLS):
            if i == 0:
                recurrent_cells.append(CELL(input_size, hidden_size))
            else:
                recurrent_cells.append(CELL(hidden_size, hidden_size))
        self.recurrent_cells = nn.ModuleList(recurrent_cells)
        # self.lstm1 = CELL(input_size, hidden_size)#RSPCell(input_size, hidden_size)
        # self.lstm2 = nn.LSTMCell(32, 32)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, inp, future=0, first_ts=0):
        outputs = []
        contexts = []
        B = inp.size(0)
        for i in range(NUM_CELLS):
            contexts.append(torch.zeros(B, self.hidden_size, dtype=torch.double, device=DEVICE))
        # c_t = torch.zeros(input.size(0), 32, dtype=torch.double)
        # h_t2 = torch.zeros(input.size(0), 32, dtype=torch.double)
        # c_t2 = torch.zeros(input.size(0), 32, dtype=torch.double)

        for i, input_t in enumerate(inp.split(1, dim=1)):
            # import pdb;pdb.set_trace()
            input_t = input_t.squeeze(dim=1)
            if len(self.embedding_frequencies) > 0:
                ts = first_ts + i
                pos_emb = get_pos_emb(ts, self.embedding_frequencies).unsqueeze(0).repeat(B, 1)
                input_t = torch.cat((input_t, pos_emb), dim=1)
            final_context = recurrent_cells_forward(self.recurrent_cells, input_t.squeeze(dim=1), contexts)
            # h_t = self.lstm1(input_t.squeeze(dim=1), h_t)#(h_t, c_t))
            # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # output = self.linear(h_t)#2)
            output = self.linear(final_context)
            outputs += [output]
        future_input = outputs[-INP_SIZE:]
        for i in range(future):  # if we should predict the future
            input_t = torch.cat(future_input, dim=1)
            if len(self.embedding_frequencies) > 0:
                ts = first_ts + inp.size(1) + i
                pos_emb = get_pos_emb(ts, self.embedding_frequencies).unsqueeze(0).repeat(B, 1)
                input_t = torch.cat((input_t, pos_emb), dim=1)
            final_context = recurrent_cells_forward(self.recurrent_cells, input_t, contexts)
            # h_t = self.lstm1(inp, h_t)#(h_t, c_t))
            # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # output = self.linear(h_t)#2)
            output = self.linear(final_context)
            outputs += [output]
            future_input.pop(0)
            future_input.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs


class LinearPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.predictor = nn.Linear(input_size, 1)

    def forward(self, inp, future=0):
        outputs = []
        # import pdb;pdb.set_trace()
        for input_t in inp.split(1, dim=1):
            output = self.predictor(input_t.squeeze(dim=1))
            outputs += [output]
        # import pdb;pdb.set_trace()
        future_input = outputs[-INP_SIZE:]
        for i in range(future):
            inp = torch.cat(future_input, dim=1)
            output = self.predictor(inp)
            outputs += [output]
            future_input.pop(0)
            future_input.append(output)
        outputs = torch.cat(outputs, dim=1)
        # import pdb;pdb.set_trace()
        return outputs


STEPS = 18
INP_SIZE = 2
HIDDEN_SIZE = 16
FUTURE = 30
LR = 0.12
MODEL = Sequence
EMB_FREQS = [np.pi / 2 ** i for i in range(1, 9)]


def sw(t: torch.Tensor) -> torch.Tensor:
    return t.unfold(dimension=1, size=INP_SIZE, step=1)


def get_train_test_data_sin():
    data = torch.load('traindata.pt')
    return (
        sw(torch.from_numpy(data[3:, :-1])).to(DEVICE),
        torch.from_numpy(data[3:, INP_SIZE:]).to(DEVICE),
        sw(torch.from_numpy(data[:3, :-1])).to(DEVICE),
        torch.from_numpy(data[:3, INP_SIZE:]).to(DEVICE),
    )


def get_train_test_data_from_ts(ts: np.ndarray, train_ratio=0.9):
    N = len(ts)
    train_size = int(N * train_ratio)
    test_size = N - train_size
    train_ts = ts[:train_size]
    test_ts = ts[train_size:]
    return (
        sw(torch.from_numpy(train_ts[:-1]).view(1, train_size - 1)).to(DEVICE),
        torch.from_numpy(train_ts[INP_SIZE:]).view(1, train_size - INP_SIZE).to(DEVICE),
        sw(torch.from_numpy(test_ts[:-1]).view(1, test_size - 1)).to(DEVICE),
        torch.from_numpy(test_ts[INP_SIZE:]).view(1, test_size - INP_SIZE).to(DEVICE)
    )


# set random seed to 0
np.random.seed(4)
torch.manual_seed(4)
# load data and make training set
data = torch.load('traindata.pt')

start = time.perf_counter_ns()

inp, target, test_input, test_target = get_train_test_data_heart_rate(heart_data)

# import pdb;pdb.set_trace()
# build the model
# seq = MODEL(INP_SIZE, HIDDEN_SIZE, EMB_FREQS).to(DEVICE).double()

cm = ContextsManager()
baseline = LinearForecaster(cm, INP_SIZE)
# INP_SIZE + 1 because correction block also takes baseline
# forecast as extra input
corrector = RSPForecaster(cm, INP_SIZE + 1, HIDDEN_SIZE)
model = RecurrentForecasterWithCorrectionBlock(cm, baseline, corrector)

criterion = nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(model.parameters(), lr=LR)
# begin to train
for i in range(STEPS):
    print('STEP: ', i)


    def closure():
        optimizer.zero_grad()
        out = run_on_inps_with_targets(model, inp, target, baseline_no_grad=False)
        loss = criterion(out, target)
        print('loss:', loss.item())
        loss.backward()
        return loss


    optimizer.step(closure)
    # begin to predict, no need to track gradient here
    with torch.no_grad():
        pred = run_on_inps_with_targets(model, test_input, test_target, baseline_no_grad=True)
        loss = criterion(pred, test_target)
        print('test loss:', loss.item())
        y = pred.detach().cpu().numpy()
    # draw the result
    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)


    def draw(yi, targeti, color):
        plt.plot(np.arange(test_input.size(1)), yi[:test_input.size(1)], color + '--', linewidth=2.0)
        plt.plot(np.arange(test_input.size(1)), targeti, color, linewidth=2.0)


    #         plt.plot(np.arange(test_input.size(1), test_input.size(1) + FUTURE), yi[test_input.size(1):],
    #                  color + ':', linewidth = 2.0)
    draw(y[0], test_target[0], 'r')
    # draw(y[1], 'g')
    # draw(y[2], 'b')
    plt.savefig('predict%d.pdf' % i)
    plt.show()

end = time.perf_counter_ns()
print(f'Elapsed time: {(end - start) / 1_000_000_000} seconds.')