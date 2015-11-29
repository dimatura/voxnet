
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

get_ipython().magic(u'namespace data np img')


# In[15]:

from sklearn import metrics as skm


# In[3]:

import voxnet


# In[5]:

from voxnet import isovox


# In[40]:

import tarfile


# In[112]:

from IPython.display import display


# In[197]:

reader = voxnet.io.NpyTarReader('/home/dmaturan/repos/voxnet/scripts/shapenet10_test.tar')


# In[7]:

out = np.load('/home/dmaturan/repos/voxnet/scripts/out.npz')


# In[10]:

yhat = out['yhat']
ygnd = out['ygnd']


# In[14]:

len(yhat), len(ygnd)


# In[21]:

from voxnet.data import shapenet10 


# In[18]:

cm = skm.confusion_matrix(ygnd, yhat)


# In[97]:

pl.imshow(cm, interpolation='nearest', cmap=pl.cm.OrRd)
pl.colorbar()
ids = sorted(map(int, shapenet10.class_id_to_name.keys()))
names = [shapenet10.class_id_to_name[str(k)] for k in ids]
ticks = np.arange(len(names))
pl.xticks(ticks, names, rotation=90)
pl.yticks(ticks, names)
pl.tight_layout()
pl.ylabel('True label')
pl.xlabel('Predicted label')


# In[153]:

ids


# In[199]:

to_name = lambda id_: shapenet10.class_id_to_name[str(id_+1)]


# In[203]:

display_ix = 12*np.random.randint(0, len(ygnd), 10)


# In[204]:

display_ix


# In[207]:

reload(isovox)


# In[ ]:




# In[205]:

reader.close()
reader = voxnet.io.NpyTarReader('/home/dmaturan/repos/voxnet/scripts/shapenet10_test.tar')

xds, yds = [], []
for ix, (xd, yd) in enumerate(reader):
    if ix in display_ix:
        dix = ix/12
        img = iv.render(xd)
        display(Image.fromarray(img))
        pred_lbl = to_name(yhat[dix])
        true_lbl = to_name(ygnd[dix])
        print ix, dix
        print("instance: {}, predicted: {}, true: {}".format(yd, pred_lbl, true_lbl))
        xds.append(xd)
        yds.append(yd)    


# In[156]:

#ix = 10895

