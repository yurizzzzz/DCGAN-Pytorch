import imageio

images = []
for e in range(60):
    img_name = './comic_result/' + str(e + 1) + '.jpg'
    images.append(imageio.imread(img_name))
imageio.mimsave('./comic_result/comic.gif', images, fps=5)