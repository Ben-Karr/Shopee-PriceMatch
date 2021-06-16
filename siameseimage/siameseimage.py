from fastai.vision.all import *

def split_df(df, label_col = 'label', val_col = 'is_valid', pct = 0.2, by_label = True, verbose = False):
    if (by_label):
        ## L-list of unique labels
        labels = L(df[label_col].unique().tolist())
        ## Randomly split labels indices
        split_labels = RandomSplitter(valid_pct= pct)(labels)  # Returns 1-pct/pct split of labels indices

        ## Mask labels to receive train/val labels instead of indices
        train_labels = labels[split_labels[0]]
        validation_labels = labels[split_labels[1]]

        ## Add colum to mark file as a part of the training/validation set
        df[val_col] = df[label_col].isin(validation_labels).astype(int)

        ## Sanity check: be shure that a label is either in the train or in the validation set
        assert((df.groupby([label_col])[val_col].nunique() > 1).sum() == 0)

    if verbose:
        total_labels, train_labels, val_labels  = len(labels), len(split_labels[0]),  len(split_labels[1])
        print(f'Total number of labels: {total_labels}; number of train/validation labels: {train_labels}/{val_labels}; relative size of validation set by labels: {val_labels / total_labels * 100}%.')
        total_size = df.shape[0]
        train_size, val_size = total_size - df[val_col].sum(), df[val_col].sum()
        print(f'Total number of instances: {total_size}, number of train/validation instances: {train_size}/{val_size}; relative size of validation set by instances: {val_size / total_size * 100}%.')

    return df

## copied from https://docs.fast.ai/tutorial.siamese.html
class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs): 
        if len(self) > 2:
            img1,img2,similarity = self
        else:
            img1,img2 = self
            similarity = 'Undetermined'
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), title=similarity, ctx=ctx, **kwargs)

## adapted from https://docs.fast.ai/tutorial.siamese.html
class SiameseTransform(Transform):
    def __init__(self, df, path, f_col = 'files', label_col = 'label', val_col = 'is_valid'):

        if (val_col not in df.columns):
            ## split the data by label and add a column to the df to capture the result
            df = split_df(df, label_col = label_col)

        ## add atributes have them available in the other methods
        self.df = df
        self.f_col = f_col
        self.label_col = label_col
        self.val_col = val_col
        self.path = Path(path)

        ## save splits as attribute, to be used by TfmdLists
        self.splits = [ (df[val_col] == 1).index.tolist(), (df[val_col] == 0).index.tolist() ]
        ## save labels as attribute
        self.labels = [ self.df[self.df[self.val_col] == i][self.label_col].unique() for i in range(2) ]

        ## Dicts of labels and containing files
        self.splbl2files = [ df[df[val_col] == i].groupby([label_col])[f_col].apply(list).to_dict() for i in range(2) ]
        ## The tuples for the validation set is drawn once 
        ## (not once per epoch as in the train case) to be comparable over different stages of training
        self.valid = {f: self._draw(f,1) for f in df[df[val_col] == 1][f_col].tolist()}
     
    def encodes(self, row):
        ## Lookup filename in df
        f = self.df.loc[row, self.f_col]
        ## if f in valid, pick f2,same = valid[f], else draw f2,same
        f2, same = self.valid.get(f, self._draw(f, 0))
        img1, img2 = PILImage.create(self.path / f), PILImage.create(self.path / f2)
        
        return SiameseImage(img1, img2, same)
    
    def _draw(self, f, split = 0):
            same = random.random() < 0.5

            ## Lookup label and split
            cls = self.df[self.df[self.f_col] == f][self.label_col].values[0]
            split = self.df[self.df[self.f_col] == f][self.val_col].values[0]

            if not same:
                ## Pick a different label than that of f
                cls = random.choice([l for l in (self.labels)[split] if l != cls])
            
            ## Pick file with chosen label
            ## Added f2 != f, since there is nothing to learn elsewise
            return random.choice([f2 for f2 in (self.splbl2files)[split][cls] if f2 != f]), int(same)


## copied from https://docs.fast.ai/tutorial.siamese.html
@typedispatch
def show_batch(x:SiameseImage, 
               y, 
               samples, 
               ctxs=None, 
               max_n=6, 
               nrows=None, 
               ncols=2, 
               figsize=None, 
               **kwargs
              ):
    if figsize is None: 
        figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: 
        ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): 
        SiameseImage(x[0][i], x[1][i], ['Not similar','Similar'][x[2][i].item()]).show(ctx=ctx)

## copied from https://docs.fast.ai/tutorial.siamese.html
@typedispatch
def show_results(x:SiameseImage, 
                 y, 
                 samples, 
                 outs, 
                 ctxs=None, 
                 max_n=6, 
                 nrows=None, 
                 ncols=2, 
                 figsize=None, 
                 **kwargs                
                ):
    if figsize is None: 
        figsize = (ncols*6, max_n//ncols * 3)
        
    if ctxs is None: 
        ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
        
    for i,ctx in enumerate(ctxs):
        title = f'Actual: {["Not similar","Similar"][int(x[2][i].item())]} \n Prediction: {["Not similar","Similar"][y[2][i].argmax().item()]}'
        SiameseImage(x[0][i], x[1][i], title).show(ctx=ctx)

## copied from https://docs.fast.ai/tutorial.siamese.html
class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head
    
    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)

## copied from https://docs.fast.ai/tutorial.siamese.html
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

## copied from https://docs.fast.ai/tutorial.siamese.html
def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())

