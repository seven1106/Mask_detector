"use strict";
const { Model } = require("sequelize");
module.exports = (sequelize, DataTypes) => {
  class markdown extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      // define association here
      markdown.belongsTo(models.User, { foreignKey: "doctorId"});

    }
  }
  markdown.init(
    {
      // id: DataTypes.INTEGER,
      contentHTML: DataTypes.TEXT({ length: "long" }),
      contentMarkdown: DataTypes.TEXT({ length: "long" }),
      description: DataTypes.TEXT({ length: "long" }),
      doctorId: DataTypes.INTEGER,
      specialtyId: DataTypes.INTEGER,
      clinicId: DataTypes.INTEGER,
    },
    {
      sequelize,
      modelName: "markdown",
    }
  );
  return markdown;
};
