class Dataset_Trajectory(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='Arrival',  # data_path 这里代表子文件夹名
                 target='Lat', scale=True, timeenc=0, freq='s', seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_folder = data_path  # 'Arrival' 或 'Departure'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dir_path = os.path.join(self.root_path, self.data_folder)
        file_list = sorted(glob.glob(os.path.join(dir_path, "*.csv")))

        # 1. 划分文件集 (按轨迹文件拆分，而不是按行拆分)
        num_files = len(file_list)
        num_train = int(num_files * 0.7)
        num_test = int(num_files * 0.2)
        num_val = num_files - num_train - num_test

        if self.set_type == 0:  # train
            files = file_list[:num_train]
        elif self.set_type == 1:  # val
            files = file_list[num_train: num_train + num_val]
        else:  # test
            files = file_list[-num_test:]

        # 2. 读取并拟合 Scaler (仅使用训练集拟合)
        if self.scale:
            train_files = file_list[:num_train]
            full_train_data = []
            for f in train_files:
                df = pd.read_csv(f)
                # 假设第一列是 Time，后面是特征
                full_train_data.append(df.iloc[:, 1:].values)
            self.scaler.fit(np.vstack(full_train_data))

        # 3. 加载当前 flag 的所有轨迹
        self.data_x = []
        self.data_stamp = []
        self.sample_indices = []  # 存储 (轨迹索引, 内部起始偏移)

        for i, f in enumerate(files):
            df_raw = pd.read_csv(f)
            # 处理特征
            df_data = df_raw.iloc[:, 1:]  # 去掉 Time 列
            if self.scale:
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            # 处理时间编码 (使用已有的 Time 列)
            df_stamp = df_raw[['Time']]
            df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.Time.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.Time.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['Time'], 1).values
            else:
                data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x.append(data)
            self.data_stamp.append(data_stamp)

            # 计算该轨迹内有效的滑动窗口起始点
            # 必须保证单条轨迹长度足够 seq_len + pred_len
            t_len = len(data)
            max_start = t_len - self.seq_len - self.pred_len + 1
            for start in range(0, max_start):
                self.sample_indices.append((i, start))

    def __getitem__(self, index):
        traj_idx, s_begin = self.sample_indices[index]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[traj_idx][s_begin:s_end]
        seq_y = self.data_x[traj_idx][r_begin:r_end]
        seq_x_mark = self.data_stamp[traj_idx][s_begin:s_end]
        seq_y_mark = self.data_stamp[traj_idx][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)