const fs = require("fs");
const path = require("path");
const webpack = require("webpack");

const runtimeDir = process.env.MELEE_LIGHT_RUNTIME_DIR;

if (!runtimeDir) {
  throw new Error("MELEE_LIGHT_RUNTIME_DIR must be set");
}

const vendorRoot = path.join(__dirname, "vendor_src");
const sourceAliases = {};
fs.readdirSync(vendorRoot, { withFileTypes: true }).forEach((entry) => {
  if (entry.isDirectory()) {
    sourceAliases[entry.name] = path.join(vendorRoot, entry.name);
  } else if (entry.isFile() && entry.name.endsWith(".js")) {
    const aliasName = path.basename(entry.name, ".js");
    if (sourceAliases[aliasName] === undefined) {
      sourceAliases[aliasName] = path.join(vendorRoot, entry.name);
    }
  }
});

module.exports = {
  cache: true,
  debug: true,
  devtool: "eval",
  entry: path.join(__dirname, "entry.js"),
  output: {
    path: path.join(runtimeDir, "js"),
    filename: "bridge.js",
  },
  resolve: {
    extensions: ["", ".js", ".json"],
    alias: sourceAliases,
    modulesDirectories: ["node_modules"],
  },
  module: {
    loaders: [
      {
        test: /\.jsx?$/,
        exclude: [/node_modules/],
        loader: "babel-loader",
        query: {
          babelrc: false,
          presets: [
            require.resolve("babel-preset-es2015"),
            require.resolve("babel-preset-stage-0"),
          ],
          plugins: [
            require.resolve("babel-plugin-transform-flow-strip-types"),
            require.resolve("babel-plugin-transform-class-properties"),
          ],
        },
      },
    ],
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env": {
        NODE_ENV: '"dev"',
      },
    }),
  ],
};
