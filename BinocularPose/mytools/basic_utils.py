from os.path import join
import os
import shutil
from .file_utils import getFileList


class ImageFolder:
    def __init__(self, path, sub=None, image='images', annot='annots',
                 no_annot=False, share_annot=False, ext='.jpg', remove_tmp=True,
                 max_per_folder=-1) -> None:
        self.root = path
        self.image = image
        self.annot = annot
        self.image_root = join(path, self.image)
        self.annot_root = join(path, self.annot)
        if not os.path.exists(self.annot_root) or no_annot:
            no_annot = True
        self.share_annot = share_annot
        self.annot_root_tmp = join(path, self.annot + '_tmp')
        if os.path.exists(self.annot_root_tmp) and remove_tmp:
            shutil.rmtree(self.annot_root_tmp)
        print('- Load data from {}'.format(path))
        if sub is None:
            print('- Try to find image names...')
            self.imgnames = getFileList(self.image_root, ext=ext, max=max_per_folder)
            print('  -> find {} images'.format(len(self.imgnames)))
            if not no_annot:
                print('- Try to find annot names...')
                self.annnames = getFileList(self.annot_root, ext='.json')
                print('  -> find {} annots'.format(len(self.annnames)))
        else:
            print('- Try to find image names of camera {}...'.format(sub))
            self.imgnames = getFileList(join(self.image_root, sub), ext=ext)
            self.imgnames = [join(sub, name) for name in self.imgnames]
            print('  -> find {} images'.format(len(self.imgnames)))
            if not no_annot:
                self.annnames = getFileList(join(self.annot_root, sub), ext='.json')
                self.annnames = [join(sub, name) for name in self.annnames]
                if len(self.annnames) != len(self.imgnames) and share_annot:
                    if len(self.annnames) == 1:
                        self.annnames = [self.annnames[0] for _ in range(len(self.imgnames))]
                else:
                    length = min(len(self.imgnames), len(self.annnames))
                    self.imgnames = self.imgnames[:length]
                    self.annnames = self.annnames[:length]
        self.isTmp = True
        self.no_annot = no_annot

    def __getitem__(self, index):
        if index > len(self.imgnames):
            print('!!! You are try to read {} image from {} images'.format(index, len(self.imgnames)))
            print('!!! Please check image path: {}'.format(self.image_root))
        imgname = join(self.image_root, self.imgnames[index])
        if self.no_annot:
            annname = None
        else:
            if self.isTmp:
                annname = join(self.annot_root_tmp, self.annnames[index])
            else:
                annname = join(self.annot_root, self.annnames[index])
        return imgname, annname

    def __len__(self):
        return len(self.imgnames)

    def __str__(self) -> str:
        return '{}: {} images'.format(self.root, len(self))
