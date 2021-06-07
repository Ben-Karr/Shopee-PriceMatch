from fastai.vision.all import *

def open_image(fname, size=224):
    img = Image.open(fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def get_split(f):
    for i,s in enumerate(splits_sets):
        if f in s:
            return i
    raise ValueError(f'File {f} is not presented in any split.')

def label_func(f):
    return df_train[df_train['image'] == f]['label_group'].values[0]

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

class SiameseTransform(Transform):
    def __init__(self, files, splits):
        self.splbl2files = [(df_train.loc[splits[i]]).groupby(['label_group'])['image'].apply(list).to_dict() for i in range(2)]
        self.valid = {f: self._draw(f,1) for f in files[splits[1]]}
        
        
    def encodes(self, f):
        f2, same = self.valid.get(f, self._draw(f, 0))
        img1, img2 = PILImage.create(f), PILImage.create(f2)
        
        return SiameseImage(img1, img2, same)
    
    def _draw(self, f, split = 0):
            same = random.random() < 0.5

            cls = label_func(f)
            split = get_split(f)

            if not same:
                cls = random.choice([l for l in labels[split] if l != cls])

            return random.choice([f2 for f2 in self.splbl2files[split][cls] if f2 != f]),int(same)

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

class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head
    
    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)

def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())