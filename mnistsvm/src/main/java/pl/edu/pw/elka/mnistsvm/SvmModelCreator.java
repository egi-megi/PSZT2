package pl.edu.pw.elka.mnistsvm;

public interface SvmModelCreator {
    SvmModel createModel();
    String modelName();
}
