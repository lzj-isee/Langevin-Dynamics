import os

main_path='./resultLD/'
contents=[]
folders=os.listdir(main_path)
folders.sort()
if 'settings.md' in folders:
    folders.remove('settings.md')
for folder in folders:
    with open(main_path+folder+'/settings.md',mode='r') as f:
        contents.append(f.readlines())
    with open(main_path+'settings.md',mode='w') as f:
        for content in contents:
            f.writelines(content)
            f.write('\n')

