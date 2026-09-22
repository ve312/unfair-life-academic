/**
 * Scales and crops an Image so it fills a width x height box centred on its
 * position without distortion, like CSS background-size: cover.
 * The image keeps its default 0.5 origin.
 * @param {Phaser.GameObjects.Image} image
 * @param {number} width
 * @param {number} height
 * @returns {Phaser.GameObjects.Image}
 */
export function coverFit(image, width, height) {
  const scale = Math.max(width / image.width, height / image.height);
  const cropWidth = width / scale;
  const cropHeight = height / scale;
  return image.setCrop((image.width - cropWidth) / 2, (image.height - cropHeight) / 2, cropWidth, cropHeight).setScale(scale);
}
