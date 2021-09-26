// Running in Google Earth Engine
// High-resolution SDD mapping
// Write by Yuan He
// 2021-08

var roi = /* color: #999900 */ee.Geometry.Polygon(
	[[[-180,-90], [-180,90], [180, 180], [180,-180]]])
// Map.centerObject(ee.Geometry.Point(120, 31),2);
Map.centerObject(roi);
var image = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select("occurrence");
// Map.addLayer(image.select("occurrence"), {min:0, max:100, palette:["red","blue"]}, "image1")
var mask = image.gt(25);
mask = mask.updateMask(mask);
mask = mask.addBands(image);

// export geotiff to asset
Export.image.toAsset({
image: mask.select(["occurrence"]),
description: "Asset-water-tif",
assetId: "Global_inland_water_tiff_30m_25occ",
scale: 30,
maxPixels:1e13
})