import numpy as np

folder = '/data/users/lkang/MP-DocVQA/imdbs/'

src = f'{folder}imdb_val.npy'

data = np.load(src, allow_pickle=True)
for item in data[1:]:
    total = item['total_doc_pages']
    item['pages'] = [0, total]
    item['image_name'] = [f"{item['image_id']}_p{ii}" for ii in range(total)]

with open(f'{folder}imdb_val_extreme_full_pages.npy', 'wb') as out:
    np.save(out, data)
