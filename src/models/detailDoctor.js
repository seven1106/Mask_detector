"use strict";
const { Model } = require("sequelize");
module.exports = (sequelize, DataTypes) => {
  class detailDoctor extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      // define association here
      detailDoctor.belongsTo(models.User, {
        foreignKey: "doctorId",
      });
      detailDoctor.belongsTo(models.allCode, {
        foreignKey: "priceId",
        targetKey: "key",
        as: "priceData",
      });
      detailDoctor.belongsTo(models.allCode, {
        targetKey: "key",
        foreignKey: "paymentId",
        as: "paymentData",
      });
      detailDoctor.belongsTo(models.allCode, {
        targetKey: "key",
        foreignKey: "provinceId",
        as: "provinceData",
      });
    }
  }
  detailDoctor.init(
    {
      // id: DataTypes.INTEGER,
      doctorId: DataTypes.INTEGER,
      specialtyId: DataTypes.INTEGER,
      priceId: DataTypes.STRING,
      paymentId: DataTypes.STRING,
      provinceId: DataTypes.STRING,
      addressClinic: DataTypes.STRING,
      nameClinic: DataTypes.STRING,
      note: DataTypes.STRING,
      count: DataTypes.INTEGER,
    },
    {
      sequelize,
      modelName: "detailDoctor",
    }
  );
  return detailDoctor;
};
