"use strict";
const { Model } = require("sequelize");
module.exports = (sequelize, DataTypes) => {
  class clinic extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      // define association here
    }
  }
  clinic.init(
    {
      // id: DataTypes.INTEGER,
      address: DataTypes.STRING,
      description: DataTypes.TEXT,
      image: DataTypes.TEXT,
      name: DataTypes.STRING,
      contentHTML: DataTypes.TEXT,
      contentMarkdown: DataTypes.TEXT,
    },
    {
      sequelize,
      modelName: "clinic",
    }
  );
  return clinic;
};
