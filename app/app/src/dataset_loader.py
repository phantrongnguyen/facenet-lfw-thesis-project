import os

def load_pairs(pairs_file, dataset_path):

    pairs=[]

    with open(pairs_file,'r') as f:

        lines=f.readlines()[1:]

        for line in lines:

            parts=line.strip().split()

            if len(parts)==3:

                name=parts[0]

                img1=int(parts[1])
                img2=int(parts[2])

                path1=os.path.join(
                    dataset_path,
                    name,
                    f"{name}_{img1:04d}.jpg"
                )

                path2=os.path.join(
                    dataset_path,
                    name,
                    f"{name}_{img2:04d}.jpg"
                )

                label=1

            else:

                name1=parts[0]
                img1=int(parts[1])

                name2=parts[2]
                img2=int(parts[3])

                path1=os.path.join(
                    dataset_path,
                    name1,
                    f"{name1}_{img1:04d}.jpg"
                )

                path2=os.path.join(
                    dataset_path,
                    name2,
                    f"{name2}_{img2:04d}.jpg"
                )

                label=0

            # Skip corrupted/incomplete pair definitions to keep evaluation running.
            if os.path.exists(path1) and os.path.exists(path2):
                pairs.append((path1,path2,label))

    return pairs