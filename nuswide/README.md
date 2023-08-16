# nuswide
This is a pytorch interface for NUSWIDE classification dataloader is also provided from  https://github.com/wenting-zhao/nuswide. Used as follows:

```
>>> import nuswide
>>> dataset = nuswide.NUSWIDEClassification('.', 'trainval')
[dataset] read ./classification_labels/classification_trainval.csv
[dataset] NUSWIDE classification set=trainval number of classes=81  number of images=134025
>>> dataset = nuswide.NUSWIDEClassification('.', 'test')
[dataset] read ./classification_labels/classification_test.csv
[dataset] NUSWIDE classification set=test number of classes=81  number of images=89470
```

You can download NUSWIDE dataset from https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html.
